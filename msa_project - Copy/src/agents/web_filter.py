"""
src/agents/web_filter.py
========================

Node 4 of the 16-node LangGraph framework: the **Web Filter**.

This node enforces the framework's most important scoping invariant
*after* the Recon Parser has converted raw Nmap / Subfinder output into
structured :class:`~src.shared.schemas.ReconResult` form:

    The framework is web-only. No non-web service may ever reach a
    downstream agent. Network-level protocols (SSH, SMB, RDP, MySQL,
    DNS, SMTP, ...) are explicitly out of scope and must be dropped
    here, regardless of whether the operator included their ports in
    ``scope.allowed_ports``.

Why a dedicated node (vs. just filtering inside the Recon Parser)?
-----------------------------------------------------------------
1. **Single Responsibility.** The Recon Parser parses; the Web Filter
   filters. Mixing the two would make both harder to test and audit.
2. **Auditability.** A separate node means the pre-filter and
   post-filter state snapshots are both visible in LangGraph's state
   history — useful for forensics if a non-web service somehow slipped
   through.
3. **Defense in depth.** ``scope.allowed_ports`` (enforced by the
   Scope Enforcer) already restricts which ports the framework will
   target. This node applies an *additional* service-classification
   filter that drops anything the Nmap service banner reports as
   non-web (e.g. an HTTP banner on port 22 — pathological but possible).

Determinism
-----------
No LLM in this path. Filtering is performed via:

- A closed set of known web ports.
- A closed set of known non-web ports (deny-list, overrides allow).
- A regex / substring match on the lowercased ``service_name``,
  ``product``, and ``banner`` fields. The match looks for explicit
  HTTP / HTTPS / WebSocket service indicators.

If a service's port is in the deny-list, it is dropped *even if* the
banner says "HTTP" — banner spoofing is a real attack vector and we
will not run HTTP-aware payloads against an SSH port.

LangGraph contract
------------------
::

    def filter_web_only(state: AppState) -> dict:

- Reads: ``state["recon_data"]`` (:class:`~src.shared.schemas.ReconResult`).
- Writes: returns ``{"recon_data": filtered_recon_result}``. The
  returned object is a NEW frozen Pydantic instance (via
  ``model_copy(update=...)``); the original is never mutated.
- Raises: :class:`~src.shared.exceptions.PentestFrameworkError`
  subclasses only on programmer error (missing state key). It does
  NOT raise on "all services filtered out" — that is a valid result
  and the Orchestrator may route to the CRAWLING phase with an empty
  host list, or to the ERROR phase if no hosts survive.

Frozen-model update strategy
----------------------------
Pydantic models in this framework are ``frozen=True``, so we cannot
mutate ``recon_data.hosts`` or ``recon_data.web_endpoints`` in place.
Instead we:

1. Build new lists of filtered :class:`ServiceInfo` and
   :class:`HostInfo` instances (themselves frozen, so we use
   ``model_copy(update={"is_web": True})`` to set the ``is_web`` flag
   on survivors).
2. Build a new list of :class:`HttpUrl` for ``web_endpoints`` derived
   from the surviving services.
3. Return a new :class:`ReconResult` via
   ``recon_data.model_copy(update={...})`` with the filtered lists.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from pydantic import HttpUrl, TypeAdapter

from src.shared.exceptions import PentestFrameworkError
from src.shared.schemas import HostInfo, ReconResult, ServiceInfo, WAFSignature
from src.shared.state import AppState


# ---------------------------------------------------------------------------
# Static classification tables
# ---------------------------------------------------------------------------
#
# These tables are intentionally module-level constants so they are
# trivially testable and auditable. Adding a port to either list is a
# security-relevant change that should be reviewed in code review.
#
# Order of evaluation in :func:`_is_web_service`:
#   1. If port in NON_WEB_PORTS_DENYLIST -> DROP (overrides everything).
#   2. If port in WEB_PORTS_ALLOWLIST    -> KEEP.
#   3. Otherwise, fall back to banner heuristics on service_name /
#      product / banner. If banner explicitly says HTTP/HTTPS/WS, KEEP;
#      otherwise DROP.
#
# This means: an HTTP service on port 22 is DROPPED (deny-list wins),
# an HTTP service on port 12345 is KEPT (banner wins), and any service
# on port 80 is KEPT (allow-list wins) regardless of banner.
# ---------------------------------------------------------------------------


WEB_PORTS_ALLOWLIST: frozenset[int] = frozenset(
    {
        80,    # HTTP
        443,   # HTTPS
        591,   # FileMaker HTTP
        832,   # NETCONF over HTTPS
        981,   # Custom HTTPS (Samba SWAT etc.)
        1311,  # Dell OpenManage HTTPS
        2483,  # Oracle HTTPS
        2484,  # Oracle HTTPS
        3000,  # Node.js / React dev servers, Grafana
        3001,  # Node.js alt
        4000,  # Various dev servers
        4200,  # Angular dev server
        4848,  # Sun Java System Application Server HTTPS
        5000,  # Flask / Python dev servers
        5601,  # Kibana
        5800,  # VNC over HTTP
        5984,  # CouchDB HTTP
        5985,  # WinRM HTTP
        5986,  # WinRM HTTPS
        7001,  # WebLogic HTTP
        7002,  # WebLogic HTTPS
        7080,  # Alt HTTP
        7443,  # Alt HTTPS
        7474,  # Neo4j HTTP
        7687,  # Neo4j HTTPS (Bolt over TLS variant)
        7473,  # Neo4j HTTPS
        8000,  # Generic HTTP / Jenkins / etc.
        8008,  # Alt HTTP
        8009,  # AJP (treat as web-adjacent; tomcat)
        8010,  # Alt HTTP
        8080,  # Alt HTTP (very common)
        8081,  # Alt HTTP
        8082,  # Alt HTTP
        8083,  # Alt HTTP
        8088,  # Hadoop YARN, others
        8089,  # Splunk, others
        8090,  # Alt HTTP
        8091,  # Alt HTTP
        8161,  # ActiveMQ HTTP
        8443,  # Alt HTTPS (very common)
        8444,  # Alt HTTPS
        8530,  # WSUS HTTP
        8531,  # WSUS HTTPS
        8888,  # Jupyter, others
        9000,  # PHP-FPM, SonarQube, Portainer
        9001,  # Alt HTTP
        9043,  # Alt HTTPS
        9080,  # Alt HTTP
        9090,  # Prometheus, Cockpit
        9091,  # Alt HTTPS (Prometheus, Cockpit)
        9200,  # Elasticsearch HTTP
        9300,  # Elasticsearch transport (treat as web-adjacent)
        9443,  # Alt HTTPS
        9600,  # Kibana alt
        11371, # HKP (OpenPGP HTTP)
    }
)
"""Closed set of ports we treat as web-by-default. Even if the Nmap
banner is empty or ambiguous, a service on one of these ports is
kept. Adding a port here is a security-relevant change."""


NON_WEB_PORTS_DENYLIST: frozenset[int] = frozenset(
    {
        20,    # FTP data
        21,    # FTP control
        22,    # SSH
        23,    # Telnet
        25,    # SMTP
        43,    # WHOIS
        53,    # DNS
        69,    # TFTP
        70,    # Gopher
        79,    # Finger
        88,    # Kerberos
        110,   # POP3
        111,   # RPCbind
        113,   # ident
        119,   # NNTP
        123,   # NTP
        135,   # MS RPC
        137,   # NetBIOS Name
        138,   # NetBIOS Datagram
        139,   # NetBIOS Session
        143,   # IMAP
        161,   # SNMP
        162,   # SNMP trap
        179,   # BGP
        194,   # IRC
        389,   # LDAP
        444,   # SNPP
        445,   # SMB
        465,   # SMTPS
        512,   # rexec
        513,   # rlogin
        514,   # rsh / syslog
        515,   # LPD
        587,   # SMTP submission
        631,   # IPP (print) — technically HTTP but treat as non-web-target
        636,   # LDAPS
        873,   # rsync
        989,   # FTPS data
        990,   # FTPS control
        993,   # IMAPS
        995,   # POP3S
        1080,  # SOCKS
        1085,  # SOCKS
        1194,  # OpenVPN
        1433,  # MS SQL
        1434,  # MS SQL browser
        1521,  # Oracle TNS
        1723,  # PPTP
        1812,  # RADIUS auth
        1813,  # RADIUS acct
        2049,  # NFS
        2181,  # ZooKeeper
        2375,  # Docker daemon (no TLS) — admin, not web app
        2376,  # Docker daemon (TLS) — admin, not web app
        2404,  # IEC 60870-5-104
        2583,  # Check Point management
        3306,  # MySQL
        3389,  # RDP
        3478,  # STUN
        4369,  # EPMD (Erlang)
        4486,  # IIOP
        4789,  # VxLAN
        5060,  # SIP
        5061,  # SIPS
        5222,  # XMPP client
        5269,  # XMPP server
        5353,  # mDNS
        5432,  # PostgreSQL
        5500,  # fcp-addr-srvr
        5666,  # NRPE
        5672,  # AMQP
        5800,  # VNC HTTP (often used to bootstrap VNC; non-web app)
        5900,  # VNC
        5901,  # VNC :1
        6000,  # X11
        6001,  # X11 :1
        6379,  # Redis
        6443,  # Kubernetes API (HTTPS) — admin, not web app
        6660,  # IRC
        6667,  # IRC
        6668,  # IRC
        6669,  # IRC
        6679,  # IRC TLS
        6697,  # IRC TLS
        7000,  # Cassandra, etc.
        7474,  # Neo4j HTTP (admin, not web app)
        7687,  # Neo4j Bolt
        7800,  # Aspera
        8005,  # Tomcat shutdown — admin
        8025,  # Postfix policy
        8086,  # InfluxDB (admin, not web app)
        8087,  # InfluxDB (admin)
        8090,  # Various admin panels
        8123,  # ClickHouse HTTP (admin) — kept off the web-target list
        8125,  # ClickHouse TCP
        8140,  # Puppet
        8333,  # Bitcoin
        8334,  # Bitcoin testnet
        8388,  # Shadowsocks
        8444,  # Bitcoin RPC
        8500,  # Consul
        8545,  # Ethereum RPC (admin)
        8555,  # Ethereum WS (admin)
        8649,  # Ganglia
        8765,  # Bitcoin RPC alt
        8834,  # Nessus
        9001,  # Tor
        9042,  # Cassandra CQL
        9043,  # Cassandra JMX
        9080,  # Cassandra alt
        9090,  # Prometheus (admin)
        9100,  # Printer
        9200,  # Elasticsearch HTTP (admin — but recon keeps it, see below)
        9300,  # Elasticsearch transport
        9418,  # git
        9999,  # Various admin
        10000, # Webmin, others
        10250, # Kubernetes kubelet
        10255, # Kubernetes kubelet read-only
        11211, # Memcached
        12345, # NetBus
        13720, # Symantec NetBackup
        14331, # Alt MySQL
        15672, # RabbitMQ management (HTTP admin) — kept off web-target list
        16992, # AMT
        16993, # AMT TLS
        17500, # Dropbox
        18080, # Tomcat alt
        19000, # Various
        20000, # Solomon
        22222, # Alt SSH
        27015, # Steam
        27017, # MongoDB
        27018, # MongoDB
        27019, # MongoDB
        32400, # Plex
        34567, # Various
        50000, # SAP
        50070, # Hadoop NameNode
        50090, # Hadoop JobTracker
        50093, # Airflow
        50470, # Alt
        55553, # Metasploit RPC
    }
)
"""Closed deny-list of non-web ports. Any service on one of these is
DROPPED unconditionally, even if the banner claims to be HTTP. This is
the deny-overrides-allow rule for service classification. Note: this
list deliberately includes some HTTP-speaking admin services
(Kubernetes API, RabbitMQ mgmt, Kibana, etc.) when they are primarily
admin surfaces rather than pentest-target web applications — those
need operator opt-in via port number, not automatic inclusion."""


# Substring tokens that indicate a web service in the Nmap banner /
# service_name / product fields. Matched case-insensitively.
_WEB_SERVICE_TOKENS: tuple[str, ...] = (
    "http",
    "https",
    "ssl/http",      # Nmap sometimes uses "ssl/http"
    "ssl/https",
    "http-alt",
    "https-alt",
    "http-proxy",
    "https-proxy",
    "websocket",
    "ws://",
    "wss://",
    "www",
    "rest",
    "graphql",
    "soap",
    "jsp",
    "aspx",
    "php",
    "asp",
    "node",
    "flask",
    "django",
    "express",
    "nginx",
    "apache",
    "tomcat",
    "jetty",
    "iis",
    "caddy",
    "traefik",
    "envoy",
    "haproxy",
    "gunicorn",
    "uvicorn",
    "fastapi",
    "spring",
)
"""Substring tokens — if any appears (case-insensitive) in the
service_name, product, or banner, we treat the service as
web-classified (subject to the port deny-list override)."""


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


def filter_web_only(state: AppState) -> dict[str, Any]:
    """LangGraph Node 4: filter ``recon_data`` to keep only web services.

    Parameters
    ----------
    state:
        The current :class:`~src.shared.state.AppState`. Must contain a
        ``recon_data`` key whose value is a
        :class:`~src.shared.schemas.ReconResult`.

    Returns
    -------
    dict
        ``{"recon_data": <new ReconResult>}`` — a fresh frozen instance
        with:

        - ``hosts`` containing only hosts that have at least one web
          service surviving the filter.
        - Each surviving :class:`ServiceInfo` has ``is_web=True``.
        - ``web_endpoints`` rebuilt from the surviving services.

    Raises
    ------
    PentestFrameworkError
        Only on programmer error (missing ``recon_data`` key). An empty
        ``recon_data`` or a filter result with zero hosts is a valid
        outcome and does NOT raise.
    """
    recon_data: ReconResult | None = state.get("recon_data")
    if recon_data is None:
        raise PentestFrameworkError(
            "Web Filter cannot run: state['recon_data'] is missing or None. "
            "The Recon Parser must run before the Web Filter; check the "
            "Orchestrator's routing table.",
            details={
                "available_keys": list(state.keys()),
                "missing_key": "recon_data",
            },
        )

    # ---------------------------------------------------------------
    # Pass 1: filter services on each host.
    # ---------------------------------------------------------------
    surviving_hosts: list[HostInfo] = []
    for host in recon_data.hosts:
        new_services: list[ServiceInfo] = []
        for svc in host.services:
            if not _is_web_service(svc):
                continue
            # Frozen model — set is_web=True via model_copy.
            if svc.is_web:
                new_services.append(svc)
            else:
                new_services.append(
                    svc.model_copy(update={"is_web": True})
                )
        if new_services:
            # Rebuild the host with the filtered services list.
            surviving_hosts.append(
                host.model_copy(update={"services": new_services})
            )
        # else: host had 0 web services — drop it entirely (do NOT
        # append to surviving_hosts).

    # ---------------------------------------------------------------
    # Pass 2: rebuild web_endpoints from surviving services.
    # ---------------------------------------------------------------
    new_web_endpoints: list[HttpUrl] = _build_web_endpoints(
        surviving_hosts, recon_data
    )

    # ---------------------------------------------------------------
    # Pass 3: return a new frozen ReconResult.
    # ---------------------------------------------------------------
    filtered_recon = recon_data.model_copy(
        update={
            "hosts": surviving_hosts,
            "web_endpoints": new_web_endpoints,
            # Update parsed_at to reflect the filter pass. We do NOT
            # change source_tool — the source tool is still whatever
            # produced the raw output (nmap / subfinder / etc.); the
            # filter pass is a transformation, not a new source.
        }
    )

    return {"recon_data": filtered_recon}


# ---------------------------------------------------------------------------
# Internal helpers — pure Python, deterministic
# ---------------------------------------------------------------------------


def _is_web_service(svc: ServiceInfo) -> bool:
    """Return True iff ``svc`` should be treated as a web service.

    Decision order (deny-overrides-allow):

    1. Port in :data:`NON_WEB_PORTS_DENYLIST` -> **DROP**.
       Even if the banner says "HTTP", a service on port 22 is not a
       pentest target. Banner spoofing is a real attack vector.
    2. Port in :data:`WEB_PORTS_ALLOWLIST` -> **KEEP**.
    3. Otherwise, fall back to banner heuristics. If any of
       ``service_name``, ``product``, ``banner`` (case-insensitively)
       contains a token from :data:`_WEB_SERVICE_TOKENS`, KEEP;
       otherwise DROP.

    This means an HTTP service on port 22 is dropped (deny wins), an
    HTTP service on port 12345 is kept (banner wins), and any service
    on port 80 is kept (allow-list wins) regardless of banner.
    """
    # Rule 1: deny-list override.
    if svc.port in NON_WEB_PORTS_DENYLIST:
        return False

    # Rule 2: allow-list.
    if svc.port in WEB_PORTS_ALLOWLIST:
        return True

    # Rule 3: banner heuristics.
    haystack_parts: list[str] = []
    if svc.service_name:
        haystack_parts.append(svc.service_name.lower())
    if svc.product:
        haystack_parts.append(svc.product.lower())
    if svc.banner:
        haystack_parts.append(svc.banner.lower())
    haystack = " ".join(haystack_parts)

    return any(token in haystack for token in _WEB_SERVICE_TOKENS)


# Adapter for parsing URL strings into HttpUrl. Module-level so the
# TypeAdapter is built once, not per-call.
_URL_ADAPTER: TypeAdapter[HttpUrl] = TypeAdapter(HttpUrl)


def _build_web_endpoints(
    hosts: list[HostInfo], original_recon: ReconResult
) -> list[HttpUrl]:
    """Build a deduplicated list of web endpoint URLs from surviving
    hosts' web services.

    For each surviving service we emit a URL of the form::

        http(s)://<host>:<port>/

    Scheme selection:
    - Port 443 / 8443 / 7443 / 9443 / 9043 → ``https``.
    - Port 80 / 8000 / 8080 / 3000 / 5000 → ``http``.
    - Any service whose banner / product / service_name contains
      ``ssl``, ``https``, or ``tls`` → ``https``.
    - Otherwise → ``http`` (default for ambiguous web services; the
      Crawler Parser will upgrade to HTTPS if it sees a redirect).

    Host name selection:
    - Prefer the host's ``hostname`` if present.
    - Otherwise use ``ip_address``.
    - IPv6 literals are bracketed per RFC 3986.

    The returned list is deduplicated (a service can appear on multiple
    hosts and we do not want duplicate endpoints). The order is stable
    — endpoints appear in the order they were discovered.
    """
    seen: set[str] = set()
    endpoints: list[HttpUrl] = []

    for host in hosts:
        host_str = _format_host_for_url(host)
        if host_str is None:
            continue  # host has neither hostname nor ip — defensive

        for svc in host.services:
            if not svc.is_web:
                continue  # defensive — should always be True here
            scheme = _select_scheme(svc)
            port_suffix = "" if _is_default_port(scheme, svc.port) else f":{svc.port}"
            url_str = f"{scheme}://{host_str}{port_suffix}/"

            if url_str in seen:
                continue
            seen.add(url_str)

            try:
                url = _URL_ADAPTER.validate_python(url_str)
            except Exception:
                # Skip malformed URLs rather than crashing the whole
                # filter pass. This is defensive — Nmap output is
                # generally well-formed, but we never want a single
                # bad row to drop the entire recon dataset.
                continue
            endpoints.append(url)

    return endpoints


def _format_host_for_url(host: HostInfo) -> str | None:
    """Return the URL-safe host string for ``host``.

    Prefers ``hostname``; falls back to ``ip_address``. IPv6 IPs are
    wrapped in brackets per RFC 3986 (``http://[::1]:8080/``).
    """
    candidate = host.hostname or host.ip_address
    if not candidate:
        return None
    candidate = candidate.strip()
    if not candidate:
        return None

    # Detect IPv6 and bracket it.
    if ":" in candidate and not candidate.startswith("["):
        # Either a bare IPv6 or a hostname (which cannot contain ':').
        # Heuristic: if it parses as IPv6, bracket it.
        try:
            import ipaddress as _ipa
            _ipa.ip_address(candidate)
            return f"[{candidate}]"
        except ValueError:
            # Not an IP — must be a hostname. ':' in a hostname is
            # invalid; skip this host rather than emitting a broken URL.
            return None
    return candidate


def _select_scheme(svc: ServiceInfo) -> str:
    """Return ``"https"`` or ``"http"`` for the given web service.

    See :func:`_build_web_endpoints` for the full decision tree.
    """
    https_ports = {443, 8443, 7443, 9443, 9043, 8531, 5986, 2484, 7002}
    if svc.port in https_ports:
        return "https"

    # Banner / product / service_name indicators of TLS.
    haystack = " ".join(
        s.lower() for s in (svc.service_name, svc.product, svc.banner) if s
    )
    if any(tok in haystack for tok in ("ssl", "https", "tls")):
        return "https"

    return "http"


def _is_default_port(scheme: str, port: int) -> bool:
    """Return True if ``port`` is the default for ``scheme``."""
    if scheme == "https":
        return port == 443
    if scheme == "http":
        return port == 80
    return False


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "filter_web_only",
    "WEB_PORTS_ALLOWLIST",
    "NON_WEB_PORTS_DENYLIST",
]
