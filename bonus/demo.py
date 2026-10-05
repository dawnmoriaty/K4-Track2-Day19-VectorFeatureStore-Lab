"""bonus/demo.py — 5-Query Demonstration Script.

Runs the 5 canonical queries from BONUS-CHALLENGE.md:
  1. Simple episodic question: "Tôi đã đọc gì về Kubernetes?"
  2. Needs profile context: "Recommend đọc gì tiếp" (needs topic_affinity)
  3. Needs fresh activity: "Tôi đang quan tâm gì gần đây?" (needs queries_last_hour)
  4. Paraphrase query: "Tài liệu về tự động mở rộng hạ tầng?" (vector wins)
  5. Mixed query: "Cho tôi summary cloud security" (episodic + profile)

Must exit 0 with all 5 contexts printed.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bonus.agent import HybridMemoryAgent


def main() -> int:
    print("=" * 60)
    print("HYBRID MEMORY AGENT — BONUS CHALLENGE DEMO")
    print("=" * 60)

    # 1. Initialize the Hybrid Memory Agent
    print("\n[Step 1] Initializing HybridMemoryAgent (Qdrant in-memory + Feast SQLite)...")
    agent = HybridMemoryAgent(
        feast_repo_path=PROJECT_ROOT / "app" / "feast_repo",
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    user_id = "u_001"
    print("Agent initialized successfully.")

    # 2. Seed realistic episodic memories for user u_001
    print("\n[Step 2] Storing episodic memories (remember) for user:", user_id)
    sample_memories = [
        "Hôm qua tôi đã đọc tài liệu về kiến trúc Kubernetes Cluster và cách triển khai Ingress Controller cho ứng dụng vi dịch vụ.",
        "Ghi chú quan trọng: Cần cấu hình Horizontal Pod Autoscaler (HPA) để tự động mở rộng số lượng Pod theo tải CPU và lượng traffic thực tế.",
        "Tài liệu bảo mật đám mây: Hướng dẫn cấu hình NetworkPolicy hạn chế truy cập mạng và mã hóa dữ liệu etcd cho Cloud Security.",
        "Đã tìm hiểu pipeline CI/CD với GitHub Actions tự động build Docker container và quét mã độc trước khi deploy lên production.",
        "Nghiên cứu về cơ sở dữ liệu phân tán: Sử dụng PostgreSQL kết hợp chỉ mục B-tree và kết nối pooling qua PgBouncer.",
    ]

    for mem in sample_memories:
        mids = agent.remember(mem, user_id=user_id)
        print(f"  + Remembered [{mids[0]}]: {mem[:65]}...")

    # 3. Execute the 5 required queries
    test_queries = [
        (
            "1. Simple Episodic Query (Vector retrieval focus)",
            "Tôi đã đọc gì về Kubernetes?",
        ),
        (
            "2. Profile-Driven Recommendation Query (Feast topic_affinity focus)",
            "Recommend đọc gì tiếp",
        ),
        (
            "3. Fresh Activity Velocity Query (Feast queries_last_hour focus)",
            "Tôi đang quan tâm gì gần đây?",
        ),
        (
            "4. Paraphrase Query (Dense vector semantic capture)",
            "Tài liệu về tự động mở rộng hạ tầng?",
        ),
        (
            "5. Mixed Query (Hybrid Memory: Episodic + Stable Profile)",
            "Cho tôi summary cloud security",
        ),
    ]

    print("\n" + "=" * 60)
    print("[Step 3] Executing 5 Demonstration Queries")
    print("=" * 60)

    for i, (title, q) in enumerate(test_queries, 1):
        print(f"\n>>> QUERY #{i}: {title}")
        context = agent.recall(query=q, user_id=user_id, top_k=2)
        print(context)

    print("\n" + "=" * 60)
    print("DEMO RUN FINISHED SUCCESSFULLY (Exit code: 0)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
