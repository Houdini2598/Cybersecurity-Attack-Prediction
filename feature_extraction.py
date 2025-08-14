# feature_extraction.py
"""
PCAP/PCAPNG → flow table (one row per unidirectional 5-tuple)
Features returned (columns used by the trainer/app):
  src_ip, dst_ip, src_port, dst_port, protocol,
  duration, packets, bytes, bytes_per_packet, packets_per_second,
  iat_mean, iat_std, tcp_syn, tcp_ack, tcp_rst, tcp_fin
"""

from collections import defaultdict
import math
import os
from typing import Dict, Tuple, Any, Iterator

import pandas as pd
import numpy as np

# Scapy imports (lazy fallback for pcapng raw reader)
from scapy.all import IP, IPv6, TCP, UDP, Ether  # type: ignore
from scapy.all import PcapReader  # type: ignore
try:
    # Raw readers for PCAPNG when PcapReader fails
    from scapy.utils import RawPcapNgReader  # type: ignore
except Exception:  # pragma: no cover
    RawPcapNgReader = None  # type: ignore


FlowKey = Tuple[str, str, int, int, str]


def _proto_label(pkt) -> str:
    if TCP in pkt:
        return "TCP"
    if UDP in pkt:
        return "UDP"
    if IP in pkt:
        # numeric IPv4 protocol number
        return str(pkt[IP].proto)
    if IPv6 in pkt:
        # numeric IPv6 next-header
        return str(pkt[IPv6].nh)
    return "OTHER"


def _flow_key(pkt) -> FlowKey | None:
    ip = pkt[IP] if IP in pkt else (pkt[IPv6] if IPv6 in pkt else None)
    if ip is None:
        return None
    proto = _proto_label(pkt)
    sport = int(pkt[TCP].sport) if TCP in pkt else (int(pkt[UDP].sport) if UDP in pkt else 0)
    dport = int(pkt[TCP].dport) if TCP in pkt else (int(pkt[UDP].dport) if UDP in pkt else 0)
    return (str(ip.src), str(ip.dst), sport, dport, proto)


def _iter_packets(path: str) -> Iterator[Any]:
    """
    Yields scapy packets from PCAP or PCAPNG.
    Tries PcapReader first; falls back to RawPcapNgReader if available.
    """
    # First try the normal streaming reader (works for pcap and many pcapng files)
    try:
        with PcapReader(path) as rd:
            for pkt in rd:
                yield pkt
        return
    except Exception:
        pass

    # If that failed and PCAPNG raw reader is present, decode frames from raw bytes
    if RawPcapNgReader is None:
        raise

    try:
        for (pkt_data, _meta) in RawPcapNgReader(path):
            try:
                yield Ether(pkt_data)
            except Exception:
                # Fall back to best-effort: skip undecodable frames
                continue
    except Exception as e:
        raise e


def pcap_to_flows(
    pcap_path: str,
    max_packets: int | None = None,
    idle_timeout: float | None = 120.0,
) -> pd.DataFrame:
    """
    Convert a PCAP/PCAPNG into a flow table.

    idle_timeout: if >0, split a flow into a new record when the gap between
    two packets in that 5-tuple exceeds this many seconds (unidirectional).
    """

    # Per-flow aggregates. We keep running stats including inter-arrival stats via Welford.
    flows: Dict[FlowKey, Dict[str, Any]] = defaultdict(lambda: {
        "start": None,
        "end": None,
        "last_ts": None,
        "pkt_count": 0,
        "byte_count": 0,
        # IAT (inter-arrival) running stats
        "iat_n": 0,
        "iat_mean": 0.0,
        "iat_M2": 0.0,
        # TCP flag counters
        "syn": 0, "ack": 0, "rst": 0, "fin": 0,
    })

    finalized_rows = []

    def finalize_flow(key: FlowKey, f: Dict[str, Any]):
        if not f["pkt_count"]:
            return
        duration = (f["end"] - f["start"]) if (f["end"] and f["start"]) else 0.0
        iat_mean = float(f["iat_mean"]) if f["iat_n"] > 0 else 0.0
        iat_var = (f["iat_M2"] / f["iat_n"]) if f["iat_n"] > 0 else 0.0
        iat_std = float(math.sqrt(max(iat_var, 0.0)))
        bpp = f["byte_count"] / f["pkt_count"]
        pps = (f["pkt_count"] / duration) if duration > 0 else float(f["pkt_count"])
        src, dst, sport, dport, proto = key
        finalized_rows.append({
            "src_ip": src,
            "dst_ip": dst,
            "src_port": int(sport),
            "dst_port": int(dport),
            "protocol": str(proto),
            "duration": float(duration),
            "packets": int(f["pkt_count"]),
            "bytes": int(f["byte_count"]),
            "bytes_per_packet": float(bpp),
            "packets_per_second": float(pps),
            "iat_mean": iat_mean,
            "iat_std": iat_std,
            "tcp_syn": int(f["syn"]),
            "tcp_ack": int(f["ack"]),
            "tcp_rst": int(f["rst"]),
            "tcp_fin": int(f["fin"]),
        })

    def update_iat(f: Dict[str, Any], t: float):
        # Welford update on gap between this packet and the previous one in the flow
        last = f["last_ts"]
        if last is None:
            f["last_ts"] = t
            return
        gap = float(t - last)
        f["last_ts"] = t
        # Guard against negative/zero timestamps from malformed traces
        if gap < 0:
            return
        n = f["iat_n"] + 1
        delta = gap - f["iat_mean"]
        mean = f["iat_mean"] + delta / n
        M2 = f["iat_M2"] + delta * (gap - mean)
        f["iat_n"] = n
        f["iat_mean"] = mean
        f["iat_M2"] = M2

    # Iterate packets
    count = 0
    for pkt in _iter_packets(pcap_path):
        count += 1
        if max_packets and count > max_packets:
            break

        # Only IP/IPv6
        if not (IP in pkt or IPv6 in pkt):
            continue

        key = _flow_key(pkt)
        if key is None:
            continue

        # Timestamp (float seconds)
        try:
            t = float(pkt.time)
        except Exception:
            # Skip if no timestamp
            continue

        length = int(len(pkt))  # approximate bytes on the wire

        f = flows[key]

        # Optional idle-timeout split
        if idle_timeout and f["end"] is not None:
            gap = t - float(f["end"])
            if gap > float(idle_timeout):
                # finalize the old flow and reset
                finalize_flow(key, f)
                flows[key] = {
                    "start": None, "end": None, "last_ts": None,
                    "pkt_count": 0, "byte_count": 0,
                    "iat_n": 0, "iat_mean": 0.0, "iat_M2": 0.0,
                    "syn": 0, "ack": 0, "rst": 0, "fin": 0,
                }
                f = flows[key]

        # Update aggregates
        if f["start"] is None:
            f["start"] = t
        f["end"] = t
        f["pkt_count"] += 1
        f["byte_count"] += length
        update_iat(f, t)

        if TCP in pkt:
            flags = int(pkt[TCP].flags)
            # Flags bits per TCP RFC: FIN(0x01) SYN(0x02) RST(0x04) PSH(0x08) ACK(0x10) URG(0x20) ECE(0x40) CWR(0x80)
            if flags & 0x02: f["syn"] += 1
            if flags & 0x10: f["ack"] += 1
            if flags & 0x04: f["rst"] += 1
            if flags & 0x01: f["fin"] += 1

    # Finalize remaining active flows
    for key, f in flows.items():
        finalize_flow(key, f)

    df = pd.DataFrame(finalized_rows)
    # Ensure all expected columns exist even if empty file
    expected = [
        "src_ip","dst_ip","src_port","dst_port","protocol",
        "duration","packets","bytes","bytes_per_packet","packets_per_second",
        "iat_mean","iat_std","tcp_syn","tcp_ack","tcp_rst","tcp_fin"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = [] if c in ("src_ip","dst_ip","protocol") else np.array([], dtype=float)

    return df[expected]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PCAP/PCAPNG → flow features CSV")
    parser.add_argument("pcap", help="Path to .pcap or .pcapng file")
    parser.add_argument("--out", default="flows.csv", help="Output CSV path")
    parser.add_argument("--max_packets", type=int, default=0, help="Limit packets (debug)")
    parser.add_argument("--idle_timeout", type=float, default=120.0, help="Split flows if idle gap > seconds (0 to disable)")
    args = parser.parse_args()

    mp = args.max_packets if args.max_packets and args.max_packets > 0 else None
    it = None if args.idle_timeout and args.idle_timeout > 0 else 0.0

    df = pcap_to_flows(args.pcap, max_packets=mp, idle_timeout=args.idle_timeout)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} flows → {os.path.abspath(args.out)}")