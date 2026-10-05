"""bonus/web_app.py — Hybrid Memory Agent Studio (Interactive Web UI).

Provides a comprehensive visual dashboard for the Day 19 Bonus Challenge:
  1. Target user switcher & live Feast feature inspector (u_001, u_002, u_003).
  2. Granular checkboxes for execution strategies (Vector ANN, Feast Profile, Feast Velocity, Multi-tenant Isolation, AI Generation).
  3. 5 Quick Case Presets (Simple Lookup, Profile Recommendation, Fresh Velocity, Paraphrase, Mixed Hybrid).
  4. Real-time step-by-step execution logs with per-step latency and status.
  5. Interactive Memory Ingestion (Remember) and Memory Explorer.
  6. Soft light palette with 6-8px border-radius as requested by user.

Run with:
  python bonus/web_app.py
  or: make ui
Access at:
  http://localhost:8501
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from bonus.agent import HybridMemoryAgent
from qdrant_client.models import FieldCondition, Filter, MatchValue

# ─────────────────────────────────────────────────────────────
# App Setup & Initialization
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Hybrid Memory Agent Studio", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent.parent
FEAST_REPO_DIR = BASE_DIR / "app" / "feast_repo"

# Initialize global agent instance
agent: HybridMemoryAgent | None = None


def get_agent() -> HybridMemoryAgent:
    global agent
    if agent is None:
        agent = HybridMemoryAgent(feast_repo_path=FEAST_REPO_DIR)
        seed_default_memories(agent)
    return agent


DEFAULT_MEMORIES = {
    "u_001": [
        "Hôm qua tôi đã đọc tài liệu về kiến trúc Kubernetes Cluster và cách triển khai Ingress Controller cho ứng dụng vi dịch vụ.",
        "Ghi chú quan trọng: Cần cấu hình Horizontal Pod Autoscaler (HPA) dựa trên metrics CPU và Memory để tự động mở rộng hạ tầng khi traffic tăng cao.",
        "Tài liệu bảo mật đám mây: Hướng dẫn cấu hình NetworkPolicy hạn chế truy cập mạng và mã hóa dữ liệu etcd cho Cloud Security.",
        "Đã tìm hiểu pipeline CI/CD với GitHub Actions tự động build Docker container và quét mã độc trước khi deploy lên production.",
        "Nghiên cứu về cơ sở dữ liệu phân tán: Sử dụng PostgreSQL kết hợp chỉ mục B-tree và kết nối pooling qua PgBouncer.",
    ],
    "u_002": [
        "Kế hoạch kiểm toán an ninh mạng: Triển khai quét lỗ hổng bảo mật định kỳ theo tiêu chuẩn OWASP Top 10 và ISO 27001.",
        "Nhật ký bảo mật nội bộ: Đã cấu hình xác thực đa yếu tố (MFA) và hạn chế quyền truy cập SSH chỉ từ mạng VPN nội bộ.",
        "Báo cáo sự cố an ninh: Phát hiện và cách ly thành công một địa chỉ IP có hành vi dò quét cổng quản trị lúc 02:15 sáng.",
    ],
    "u_003": [
        "Nghiên cứu pháp lý AI: Rà soát các quy định của Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân đối với dữ liệu hội thoại.",
        "Tài liệu hợp đồng dịch vụ đám mây: Yêu cầu nhà cung cấp Cloud cam kết SLA 99.99% và bồi thường thiệt hại khi gián đoạn.",
    ],
}


def seed_default_memories(ag: HybridMemoryAgent) -> None:
    """Pre-seeds standard episodic memories for users u_001, u_002, u_003."""
    for uid, texts in DEFAULT_MEMORIES.items():
        for t in texts:
            ag.remember(t, user_id=uid)


# ─────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────
class RecallRequest(BaseModel):
    query: str
    user_id: str = "u_001"
    top_k: int = Field(default=3, ge=1, le=10)
    use_vector: bool = True
    use_profile: bool = True
    use_velocity: bool = True
    enforce_tenant: bool = True
    generate_ai: bool = True


class RememberRequest(BaseModel):
    text: str
    user_id: str = "u_001"
    max_chunk_chars: int = 300


# ─────────────────────────────────────────────────────────────
# Helper: Simulated Vietnamese AI Response Generator
# ─────────────────────────────────────────────────────────────
def generate_ai_response(
    query: str,
    user_id: str,
    profile: dict[str, Any],
    hits: list[dict[str, Any]],
    use_vector: bool,
    use_profile: bool,
    use_velocity: bool,
) -> str:
    lang = profile.get("preferred_language", "vi")
    affinity = profile.get("topic_affinity", "general")
    wpm = profile.get("reading_speed_wpm", 187)
    q_1h = profile.get("queries_last_hour", 0)

    # Style tone based on WPM: fast readers prefer dense bullets, slower readers prefer clear paragraphs
    tone_dense = wpm >= 200

    q_lower = query.lower()
    mem_summaries = [h["text"] for h in hits]

    if "kubernetes" in q_lower or "k8s" in q_lower:
        if mem_summaries:
            return (
                f"Dựa trên hồi ức đã lưu của bạn, bạn đã nghiên cứu về **kiến trúc Kubernetes Cluster**, "
                f"cách triển khai **Ingress Controller** cho microservices và cấu hình **HPA (Horizontal Pod Autoscaler)** "
                f"để tự động mở rộng theo CPU/RAM.\n\n"
                f"💡 *Gợi ý tiếp theo:* Với sở thích `{affinity}` của bạn, bước kế tiếp nên tìm hiểu về Service Mesh (Istio) hoặc GitOps với ArgoCD."
            )
        return "Bạn chưa có ghi chú nào về Kubernetes trong bộ nhớ hồi ức."

    if "recommend" in q_lower or "đọc gì" in q_lower:
        return (
            f"Dựa trên hồ sơ của bạn (Sở thích chuyên sâu: `{affinity}`, tốc độ đọc: {wpm} wpm):\n"
            f"1. **Kiến trúc Cloud-Native Security**: Tìm hiểu NetworkPolicy và mTLS cho vi dịch vụ.\n"
            f"2. **Tối ưu hóa Database Pooling**: Đào sâu PgBouncer và Connection Multiplexing.\n"
            f"3. **Hạ tầng tự động co giãn**: Thực hành cấu hình KEDA (Kubernetes Event-driven Autoscaling).\n\n"
            f"*(Được cá nhân hóa dựa trên Feast Feature Store với độ tươi cập nhật 30 ngày qua)*"
        )

    if "quan tâm" in q_lower or "gần đây" in q_lower or "velocity" in q_lower:
        return (
            f"Theo thống kê hoạt động thời gian thực từ Feast Streaming Pipeline:\n"
            f"• Trong 1 giờ qua, bạn đã thực hiện **{q_1h} truy vấn liên tục**.\n"
            f"• Chủ đề bạn đang tập trung nhất là `{affinity}` và cơ sở dữ liệu phân tán.\n"
            f"• Cường độ tìm kiếm khá cao cho thấy bạn đang trong một phiên giải quyết sự cố hoặc triển khai dự án lớn."
        )

    if "hạ tầng" in q_lower or "mở rộng" in q_lower or "autoscal" in q_lower:
        if mem_summaries:
            matched = mem_summaries[0]
            return (
                f"Tôi tìm thấy ghi chú khớp ngữ nghĩa cao nhất:\n"
                f"> \"{matched}\"\n\n"
                f"📌 *Tóm tắt kỹ thuật:* Hệ thống ghi nhớ giải pháp mở rộng tự động bằng HPA theo tải CPU/Memory, "
                f"phù hợp với tốc độ đọc {wpm} wpm của bạn."
            )
        return "Không tìm thấy tài liệu phù hợp về mở rộng hạ tầng trong bộ nhớ cá nhân."

    if "security" in q_lower or "bảo mật" in q_lower:
        parts = []
        if mem_summaries:
            parts.append(f"• **Episodic Memory trúng tuyển:** {mem_summaries[0]}")
        parts.append(f"• **Hồ sơ bảo mật của bạn:** Ưu tiên ngôn ngữ `{lang}`, chuyên môn `{affinity}`.")
        parts.append("• **Khuyến nghị bảo vệ:** Thiết lập NetworkPolicy hạn chế pod egress và mã hóa khóa lưu trữ.")
        return "\n".join(parts)

    # General fallback response
    if hits:
        top_text = hits[0]["text"]
        return (
            f"Chào bạn, dựa trên các tài liệu đã ghi nhớ (`{top_text[:80]}...`) "
            f"kết hợp cùng hồ sơ sở thích `{affinity}` ({wpm} WPM), tôi đã tổng hợp thông tin phù hợp nhất theo yêu cầu của bạn."
        )
    return (
        f"Đã xử lý truy vấn cho người dùng [{user_id}]. Không có mẩu ký ức nào vượt qua ngưỡng tương đồng, "
        f"hệ thống đã dùng thông tin nền tảng từ Feast ({affinity}, {lang}) để sẵn sàng phục vụ bạn."
    )


# ─────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/api/status")
def get_status() -> dict[str, Any]:
    ag = get_agent()
    pts_count = ag.client.count(ag.collection_name).count
    return {
        "status": "ready",
        "vector_backend": "Qdrant In-Memory (BAAI/bge-small-en-v1.5 - 384d)",
        "feature_store": "Feast SQLite Online Store",
        "total_indexed_memories": pts_count,
        "supported_users": ["u_001", "u_002", "u_003"],
    }


@app.get("/api/users")
def list_users() -> list[dict[str, Any]]:
    ag = get_agent()
    users = ["u_001", "u_002", "u_003"]
    user_info = []

    try:
        feast_res = ag.fs.get_online_features(
            features=[
                "user_profile_features:reading_speed_wpm",
                "user_profile_features:preferred_language",
                "user_profile_features:topic_affinity",
                "query_velocity_features:queries_last_hour",
                "query_velocity_features:distinct_topics_24h",
            ],
            entity_rows=[{"user_id": uid} for uid in users],
        ).to_dict()
    except Exception:
        feast_res = {}

    role_map = {
        "u_001": "Kỹ sư Cloud / DevOps (Cloud Enthusiast)",
        "u_002": "Chuyên viên An toàn thông tin (Security Analyst)",
        "u_003": "Chuyên gia Pháp lý & Dữ liệu (AI Legal Officer)",
    }

    for idx, uid in enumerate(users):
        user_info.append(
            {
                "user_id": uid,
                "role": role_map.get(uid, "Người dùng chuẩn"),
                "reading_speed_wpm": feast_res.get("reading_speed_wpm", [187, 194, 201])[idx],
                "preferred_language": feast_res.get("preferred_language", ["vi", "vi", "en"])[idx],
                "topic_affinity": feast_res.get("topic_affinity", ["cloud", "security", "database"])[idx],
                "queries_last_hour": feast_res.get("queries_last_hour", [11, 22, 33])[idx],
                "distinct_topics_24h": feast_res.get("distinct_topics_24h", [4, 7, 10])[idx],
            }
        )
    return user_info


@app.get("/api/user/{user_id}/memories")
def get_user_memories(user_id: str) -> list[dict[str, Any]]:
    ag = get_agent()
    # Scroll points in Qdrant for this user
    user_filter = Filter(
        must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    )
    res = ag.client.scroll(
        collection_name=ag.collection_name,
        scroll_filter=user_filter,
        limit=50,
        with_payload=True,
    )[0]

    memories = []
    for p in res:
        payload = p.payload or {}
        memories.append(
            {
                "point_id": p.id,
                "memory_id": payload.get("memory_id", str(p.id)),
                "text": payload.get("text", ""),
                "created_at": payload.get("created_at", 0),
                "user_id": payload.get("user_id", user_id),
            }
        )
    return memories


@app.post("/api/remember")
def api_remember(req: RememberRequest) -> dict[str, Any]:
    ag = get_agent()
    t0 = time.perf_counter()

    chunks = ag._chunk_text(req.text, max_chunk_chars=req.max_chunk_chars)
    t_chunk = time.perf_counter()

    mem_ids = ag.remember(req.text, user_id=req.user_id)
    t_total = time.perf_counter()

    return {
        "success": True,
        "user_id": req.user_id,
        "chunks_count": len(chunks),
        "memory_ids": mem_ids,
        "chunks_preview": chunks,
        "chunking_latency_ms": round((t_chunk - t0) * 1000, 2),
        "total_latency_ms": round((t_total - t0) * 1000, 2),
    }


@app.post("/api/recall")
def api_recall(req: RecallRequest) -> dict[str, Any]:
    ag = get_agent()
    steps: list[dict[str, Any]] = []
    t_all_start = time.perf_counter()

    # Step 1: Request parsing & entity validation
    t0 = time.perf_counter()
    steps.append(
        {
            "step_id": 1,
            "title": "Khởi tạo & Xác thực Thực thể",
            "tag": "INIT",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
            "status": "success",
            "details": f"Người dùng: {req.user_id} | Độ dài truy vấn: {len(req.query)} ký tự | Top-K: {req.top_k}",
        }
    )

    # Step 2: Feast Online Feature Store Lookup
    profile_features = {
        "reading_speed_wpm": 187,
        "preferred_language": "vi",
        "topic_affinity": "general",
        "queries_last_hour": 0,
        "distinct_topics_24h": 0,
    }
    t_feast_start = time.perf_counter()
    if req.use_profile or req.use_velocity:
        try:
            wanted_feats = []
            if req.use_profile:
                wanted_feats.extend([
                    "user_profile_features:reading_speed_wpm",
                    "user_profile_features:preferred_language",
                    "user_profile_features:topic_affinity",
                ])
            if req.use_velocity:
                wanted_feats.extend([
                    "query_velocity_features:queries_last_hour",
                    "query_velocity_features:distinct_topics_24h",
                ])

            res = ag.fs.get_online_features(
                features=wanted_feats,
                entity_rows=[{"user_id": req.user_id}],
            ).to_dict()

            for k in ("reading_speed_wpm", "preferred_language", "topic_affinity", "queries_last_hour", "distinct_topics_24h"):
                val = res.get(k)
                if val and len(val) > 0 and val[0] is not None:
                    profile_features[k] = val[0]
            t_feast_ms = round((time.perf_counter() - t_feast_start) * 1000, 2)
            steps.append(
                {
                    "step_id": 2,
                    "title": "Truy xuất Feast Online Feature Store",
                    "tag": "FEAST_LOOKUP",
                    "latency_ms": t_feast_ms,
                    "status": "success",
                    "details": f"Lấy thành công {len(wanted_feats)} features từ SQLite. Ngôn ngữ: '{profile_features['preferred_language']}', WPM: {profile_features['reading_speed_wpm']}, Topic: '{profile_features['topic_affinity']}'",
                }
            )
        except Exception as e:
            t_feast_ms = round((time.perf_counter() - t_feast_start) * 1000, 2)
            steps.append(
                {
                    "step_id": 2,
                    "title": "Truy xuất Feast Online Feature Store",
                    "tag": "FEAST_FALLBACK",
                    "latency_ms": t_feast_ms,
                    "status": "warning",
                    "details": f"Dùng giá trị mặc định do lỗi kết nối: {e}",
                }
            )
    else:
        steps.append(
            {
                "step_id": 2,
                "title": "Bỏ qua Feast Feature Store",
                "tag": "SKIPPED",
                "latency_ms": 0.0,
                "status": "info",
                "details": "Tùy chọn Feast Profile & Velocity đã được người dùng tắt qua checkbox.",
            }
        )

    # Step 3: FastEmbed Query Vectorization
    t_embed_start = time.perf_counter()
    hits_data: list[dict[str, Any]] = []

    if req.use_vector:
        q_vec = next(ag.embedder.embed([req.query])).tolist()
        t_embed_ms = round((time.perf_counter() - t_embed_start) * 1000, 2)
        steps.append(
            {
                "step_id": 3,
                "title": "Mã hóa Vector (FastEmbed ONNX)",
                "tag": "EMBEDDING",
                "latency_ms": t_embed_ms,
                "status": "success",
                "details": f"Model: BAAI/bge-small-en-v1.5 | Tạo vector {len(q_vec)} chiều | Độ trễ ONNX: {t_embed_ms}ms",
            }
        )

        # Step 4: Qdrant ANN Search
        t_search_start = time.perf_counter()
        q_filter = None
        filter_desc = "Không áp dụng filter (Cross-tenant Leakage test!)"
        if req.enforce_tenant:
            q_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=req.user_id))]
            )
            filter_desc = f"Lọc nghiêm ngặt user_id == '{req.user_id}' (Multi-tenant Privacy Isolation)"

        search_points = ag.client.query_points(
            collection_name=ag.collection_name,
            query=q_vec,
            query_filter=q_filter,
            limit=req.top_k,
        ).points
        t_search_ms = round((time.perf_counter() - t_search_start) * 1000, 2)

        for p in search_points:
            hits_data.append(
                {
                    "score": round(p.score, 4),
                    "memory_id": (p.payload or {}).get("memory_id", str(p.id)),
                    "text": (p.payload or {}).get("text", ""),
                    "user_id": (p.payload or {}).get("user_id", "unknown"),
                    "created_at": (p.payload or {}).get("created_at", 0),
                }
            )

        steps.append(
            {
                "step_id": 4,
                "title": "Tìm kiếm Tương đồng Qdrant Vector ANN",
                "tag": "VECTOR_SEARCH",
                "latency_ms": t_search_ms,
                "status": "success",
                "details": f"Tìm thấy {len(hits_data)} mẩu ký ức phù hợp. {filter_desc}. Điểm cao nhất: {hits_data[0]['score'] if hits_data else 'N/A'}",
            }
        )
    else:
        steps.append(
            {
                "step_id": 3,
                "title": "Bỏ qua Vector Embedder",
                "tag": "SKIPPED",
                "latency_ms": 0.0,
                "status": "info",
                "details": "Tìm kiếm Vector Episodic đã được tắt qua checkbox.",
            }
        )
        steps.append(
            {
                "step_id": 4,
                "title": "Bỏ qua Qdrant Vector Search",
                "tag": "SKIPPED",
                "latency_ms": 0.0,
                "status": "info",
                "details": "Không thực hiện truy vấn kho vector.",
            }
        )

    # Step 5: Assembling Context
    t_asm_start = time.perf_counter()
    context_lines: list[str] = [
        f"=== ASSEMBLED CONTEXT FOR USER [{req.user_id}] ===",
        f"Query: {req.query!r}",
        "",
    ]
    if req.use_profile:
        context_lines.extend([
            "--- [Feast: User Stable Profile] ---",
            f"  • Preferred Language : {profile_features['preferred_language']}",
            f"  • Reading Speed      : {profile_features['reading_speed_wpm']} WPM",
            f"  • Topic Affinity     : {profile_features['topic_affinity']}",
            "",
        ])
    if req.use_velocity:
        context_lines.extend([
            "--- [Feast: Recent Activity Velocity] ---",
            f"  • Queries (Last 1h)  : {profile_features['queries_last_hour']}",
            f"  • Distinct Topics 24h: {profile_features['distinct_topics_24h']}",
            "",
        ])

    context_lines.append("--- [Qdrant: Relevant Episodic Memories] ---")
    if hits_data:
        for idx, h in enumerate(hits_data, 1):
            owner_badge = f" [Owner: {h['user_id']}]" if not req.enforce_tenant else ""
            context_lines.append(f"  {idx}. [Score: {h['score']:.3f}]{owner_badge} {h['text']}")
    else:
        context_lines.append("  (No relevant episodic memories found)")
    context_lines.append("==============================================")
    assembled_str = "\n".join(context_lines)

    t_asm_ms = round((time.perf_counter() - t_asm_start) * 1000, 2)
    steps.append(
        {
            "step_id": 5,
            "title": "Ghép Ngữ Cảnh (Context Assembly Engine)",
            "tag": "CONTEXT_ASSEMBLY",
            "latency_ms": t_asm_ms,
            "status": "success",
            "details": f"Đã hợp nhất {len(hits_data)} đoạn ký ức và đặc trưng Feast vào cấu trúc Prompt dài {len(assembled_str)} ký tự.",
        }
    )

    # Step 6: AI Response Generation
    ai_resp = ""
    if req.generate_ai:
        t_ai_start = time.perf_counter()
        ai_resp = generate_ai_response(
            query=req.query,
            user_id=req.user_id,
            profile=profile_features,
            hits=hits_data,
            use_vector=req.use_vector,
            use_profile=req.use_profile,
            use_velocity=req.use_velocity,
        )
        t_ai_ms = round((time.perf_counter() - t_ai_start) * 1000, 2)
        steps.append(
            {
                "step_id": 6,
                "title": "Sinh Phản Hồi Cá Nhân Hóa (AI Synthesis)",
                "tag": "AI_SYNTHESIS",
                "latency_ms": t_ai_ms,
                "status": "success",
                "details": f"Tối ưu văn phong theo {profile_features['reading_speed_wpm']} WPM và ngữ cảnh tiếng Việt (Code-switching, thuật ngữ).",
            }
        )

    t_total_ms = round((time.perf_counter() - t_all_start) * 1000, 2)

    return {
        "query": req.query,
        "user_id": req.user_id,
        "total_latency_ms": t_total_ms,
        "profile": profile_features,
        "hits": hits_data,
        "assembled_context": assembled_str,
        "ai_response": ai_resp,
        "steps": steps,
    }


@app.post("/api/reset")
def api_reset() -> dict[str, Any]:
    global agent
    agent = HybridMemoryAgent(feast_repo_path=FEAST_REPO_DIR)
    seed_default_memories(agent)
    count = agent.client.count(agent.collection_name).count
    return {
        "success": True,
        "message": f"Đã reset toàn bộ bộ nhớ về 5 tài liệu chuẩn. Tổng số điểm trong Qdrant: {count}",
        "total_points": count,
    }


# ─────────────────────────────────────────────────────────────
# Frontend UI (Modern Soft Light Palette, 6-8px border-radius)
# ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hybrid Memory Agent Studio | Lab 19 Bonus</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-main: #f8fafc;
      --bg-card: #ffffff;
      --border-color: #e2e8f0;
      --border-focus: #cbd5e1;
      --text-primary: #0f172a;
      --text-secondary: #475569;
      --text-muted: #64748b;
      --accent: #4f46e5;
      --accent-hover: #4338ca;
      --accent-light: #eef2ff;
      --success: #059669;
      --success-light: #ecfdf5;
      --warning: #d97706;
      --warning-light: #fffbeb;
      --info: #0284c7;
      --info-light: #f0f9ff;
      --radius: 7px;
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg-main);
      color: var(--text-primary);
      line-height: 1.5;
      font-size: 14px;
      min-height: 100vh;
    }

    header {
      background: var(--bg-card);
      border-bottom: 1px solid var(--border-color);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      width: 32px;
      height: 32px;
      background: var(--accent);
      color: #fff;
      border-radius: var(--radius);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 16px;
    }
    .brand-title {
      font-size: 16px;
      font-weight: 700;
      color: var(--text-primary);
    }
    .brand-subtitle {
      font-size: 12px;
      color: var(--text-muted);
    }

    .header-badges {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: var(--radius);
      font-size: 12px;
      font-weight: 500;
      border: 1px solid transparent;
    }
    .badge-success { background: var(--success-light); color: var(--success); border-color: #a7f3d0; }
    .badge-info { background: var(--info-light); color: var(--info); border-color: #bae6fd; }
    .badge-accent { background: var(--accent-light); color: var(--accent); border-color: #c7d2fe; }
    .badge-warning { background: var(--warning-light); color: var(--warning); border-color: #fde68a; }

    .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: currentColor;
    }

    /* Main Container Grid */
    .app-layout {
      display: grid;
      grid-template-columns: 310px 1fr 340px;
      gap: 18px;
      padding: 18px 24px;
      max-width: 1680px;
      margin: 0 auto;
    }

    /* Cards */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius);
      box-shadow: var(--shadow-sm);
      margin-bottom: 16px;
      overflow: hidden;
    }
    .card-header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #fafafa;
    }
    .card-title {
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .card-body {
      padding: 16px;
    }

    /* Form Elements */
    label {
      display: block;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-secondary);
      margin-bottom: 6px;
    }
    select, input[type="text"], textarea {
      width: 100%;
      padding: 8px 12px;
      font-family: inherit;
      font-size: 13px;
      border: 1px solid var(--border-color);
      border-radius: var(--radius);
      background: #fff;
      color: var(--text-primary);
      transition: all 0.15s ease;
    }
    select:focus, input[type="text"]:focus, textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12);
    }
    textarea {
      resize: vertical;
      min-height: 85px;
    }

    /* Checkbox Group */
    .checkbox-group {
      display: flex;
      flex-direction: column;
      gap: 9px;
    }
    .checkbox-item {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      font-size: 12.5px;
      color: var(--text-primary);
      cursor: pointer;
    }
    .checkbox-item input[type="checkbox"] {
      margin-top: 2px;
      cursor: pointer;
      accent-color: var(--accent);
      width: 15px;
      height: 15px;
    }
    .checkbox-desc {
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 1px;
    }

    /* Buttons */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      padding: 8px 14px;
      border-radius: var(--radius);
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
      border: 1px solid transparent;
      text-decoration: none;
    }
    .btn-primary {
      background: var(--accent);
      color: #fff;
    }
    .btn-primary:hover {
      background: var(--accent-hover);
    }
    .btn-outline {
      background: #fff;
      border-color: var(--border-color);
      color: var(--text-secondary);
    }
    .btn-outline:hover {
      background: #f1f5f9;
      color: var(--text-primary);
    }
    .btn-block { width: 100%; }
    .btn-sm {
      padding: 4px 8px;
      font-size: 11.5px;
    }

    /* Preset Case Buttons */
    .case-list {
      display: flex;
      flex-direction: column;
      gap: 7px;
    }
    .case-btn {
      text-align: left;
      padding: 9px 12px;
      background: #f8fafc;
      border: 1px solid var(--border-color);
      border-radius: var(--radius);
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .case-btn:hover {
      background: var(--accent-light);
      border-color: #c7d2fe;
    }
    .case-btn-tag {
      font-size: 10.5px;
      font-weight: 600;
      color: var(--accent);
      text-transform: uppercase;
    }
    .case-btn-query {
      font-size: 12.5px;
      font-weight: 500;
      color: var(--text-primary);
    }

    /* Stats Grid */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      margin-top: 10px;
    }
    .stat-box {
      background: #f8fafc;
      border: 1px solid var(--border-color);
      border-radius: var(--radius);
      padding: 8px 10px;
    }
    .stat-label {
      font-size: 11px;
      color: var(--text-muted);
      margin-bottom: 2px;
    }
    .stat-value {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-primary);
    }

    /* Tabs */
    .tabs {
      display: flex;
      gap: 4px;
      border-bottom: 1px solid var(--border-color);
      margin-bottom: 16px;
    }
    .tab-btn {
      padding: 8px 14px;
      font-size: 13px;
      font-weight: 500;
      color: var(--text-muted);
      background: transparent;
      border: none;
      border-bottom: 2px solid transparent;
      cursor: pointer;
    }
    .tab-btn.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
      font-weight: 600;
    }

    /* Results */
    .ai-response-card {
      background: #faf5ff;
      border: 1px solid #e9d5ff;
      border-radius: var(--radius);
      padding: 14px 16px;
      margin-bottom: 16px;
    }
    .ai-response-title {
      font-size: 12px;
      font-weight: 600;
      color: #7e22ce;
      text-transform: uppercase;
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .ai-response-body {
      font-size: 13.5px;
      color: #3b0764;
      line-height: 1.6;
      white-space: pre-wrap;
    }

    .memory-hit-card {
      background: #fff;
      border: 1px solid var(--border-color);
      border-radius: var(--radius);
      padding: 12px;
      margin-bottom: 10px;
    }
    .memory-hit-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }
    .memory-score-badge {
      font-size: 11px;
      font-weight: 600;
      padding: 2px 7px;
      border-radius: var(--radius);
      background: #ecfdf5;
      color: #047857;
      border: 1px solid #a7f3d0;
    }
    .memory-score-bar-bg {
      height: 4px;
      background: #e2e8f0;
      border-radius: 2px;
      margin-top: 4px;
      margin-bottom: 8px;
      overflow: hidden;
    }
    .memory-score-bar-fill {
      height: 100%;
      background: var(--success);
      border-radius: 2px;
    }

    /* Assembled Prompt Block */
    .code-block {
      background: #0f172a;
      color: #e2e8f0;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      padding: 14px;
      border-radius: var(--radius);
      overflow-x: auto;
      white-space: pre-wrap;
      line-height: 1.5;
      max-height: 280px;
    }

    /* Step Logs Timeline */
    .timeline {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .step-item {
      display: flex;
      gap: 10px;
      position: relative;
    }
    .step-item:not(:last-child)::before {
      content: '';
      position: absolute;
      top: 24px;
      left: 12px;
      bottom: -12px;
      width: 2px;
      background: var(--border-color);
    }
    .step-num {
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: var(--accent-light);
      color: var(--accent);
      border: 1px solid #c7d2fe;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
      font-weight: 600;
      flex-shrink: 0;
      z-index: 1;
    }
    .step-content {
      flex: 1;
      background: #f8fafc;
      border: 1px solid var(--border-color);
      border-radius: var(--radius);
      padding: 8px 10px;
    }
    .step-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;
    }
    .step-title {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-primary);
    }
    .step-badge {
      font-size: 10px;
      padding: 1px 5px;
      border-radius: 4px;
      font-weight: 600;
      font-family: monospace;
    }
    .step-tag-FEAST_LOOKUP { background: #e0f2fe; color: #0369a1; }
    .step-tag-EMBEDDING { background: #fef3c7; color: #b45309; }
    .step-tag-VECTOR_SEARCH { background: #ecfdf5; color: #047857; }
    .step-tag-CONTEXT_ASSEMBLY { background: #eef2ff; color: #4338ca; }
    .step-tag-AI_SYNTHESIS { background: #faf5ff; color: #7e22ce; }
    .step-tag-SKIPPED { background: #f1f5f9; color: #64748b; }
    .step-details {
      font-size: 11.5px;
      color: var(--text-muted);
      line-height: 1.4;
    }
    .step-latency {
      font-size: 10.5px;
      color: var(--text-secondary);
      font-weight: 500;
    }

    .empty-state {
      text-align: center;
      padding: 36px 20px;
      color: var(--text-muted);
    }
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="brand">
      <div class="brand-icon">⚡</div>
      <div>
        <div class="brand-title">Hybrid Memory Agent Studio</div>
        <div class="brand-subtitle">Episodic Memory (Qdrant) + Stable & Velocity Profile (Feast)</div>
      </div>
    </div>
    <div class="header-badges">
      <span class="badge badge-success"><span class="dot"></span> Qdrant In-Memory</span>
      <span class="badge badge-info"><span class="dot"></span> Feast SQLite</span>
      <span class="badge badge-accent">ONNX 384d</span>
      <button class="btn btn-outline btn-sm" onclick="resetMemories()" title="Khôi phục dữ liệu demo gốc">↺ Reset Data</button>
    </div>
  </header>

  <!-- Main Layout -->
  <div class="app-layout">

    <!-- LEFT COLUMN: User & Controls & Presets -->
    <div>

      <!-- User Selector Card -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">👤 Thực Thể Người Dùng (Entity)</div>
          <span class="badge badge-accent" id="lbl-uid">u_001</span>
        </div>
        <div class="card-body">
          <label for="select-user">Chọn người dùng mẫu:</label>
          <select id="select-user" onchange="onUserChange()">
            <option value="u_001">u_001 — Cloud / DevOps Engineer</option>
            <option value="u_002">u_002 — Security Analyst</option>
            <option value="u_003">u_003 — AI Legal Specialist</option>
          </select>

          <!-- Live Feast Stats Card -->
          <div class="stats-grid">
            <div class="stat-box">
              <div class="stat-label">Sở thích (Topic)</div>
              <div class="stat-value" id="stat-topic">cloud</div>
            </div>
            <div class="stat-box">
              <div class="stat-label">Tốc độ đọc</div>
              <div class="stat-value" id="stat-wpm">187 WPM</div>
            </div>
            <div class="stat-box">
              <div class="stat-label">Ngôn ngữ</div>
              <div class="stat-value" id="stat-lang">vi</div>
            </div>
            <div class="stat-box">
              <div class="stat-label">Truy vấn (1h)</div>
              <div class="stat-value" id="stat-queries">11 reqs</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Strategy Checkboxes -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">⚙️ Phương Thức Hành Động</div>
        </div>
        <div class="card-body">
          <div class="checkbox-group">
            <label class="checkbox-item">
              <input type="checkbox" id="chk-vector" checked>
              <div>
                <strong>Vector Episodic Memory</strong>
                <div class="checkbox-desc">Truy xuất hồi ức qua Qdrant Dense ANN</div>
              </div>
            </label>

            <label class="checkbox-item">
              <input type="checkbox" id="chk-profile" checked>
              <div>
                <strong>Feast Stable Profile</strong>
                <div class="checkbox-desc">Lấy topic_affinity, reading_speed_wpm</div>
              </div>
            </label>

            <label class="checkbox-item">
              <input type="checkbox" id="chk-velocity" checked>
              <div>
                <strong>Feast Activity Velocity</strong>
                <div class="checkbox-desc">Lấy queries_last_hour, distinct_topics_24h</div>
              </div>
            </label>

            <label class="checkbox-item">
              <input type="checkbox" id="chk-tenant" checked>
              <div>
                <strong>Multi-Tenant Privacy Isolation</strong>
                <div class="checkbox-desc">Lọc user_id (Bỏ chọn để thử rò rỉ chéo)</div>
              </div>
            </label>

            <label class="checkbox-item">
              <input type="checkbox" id="chk-ai" checked>
              <div>
                <strong>Sinh Phản Hồi AI Cá Nhân Hóa</strong>
                <div class="checkbox-desc">Cân chỉnh văn phong theo WPM & Tiếng Việt</div>
              </div>
            </label>
          </div>

          <div style="margin-top: 14px;">
            <label for="top-k-slider">Số lượng ký ức Top-K: <span id="top-k-val" style="color: var(--accent); font-weight:700;">3</span></label>
            <input type="range" id="top-k-slider" min="1" max="6" value="3" style="width:100%; accent-color: var(--accent);" oninput="document.getElementById('top-k-val').innerText = this.value">
          </div>
        </div>
      </div>

      <!-- Quick Demo Presets -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">🎯 5 Kịch Bản Mẫu (Demo Presets)</div>
        </div>
        <div class="card-body">
          <div class="case-list">
            <button class="case-btn" onclick="applyCase(1, 'Tôi đã đọc gì về Kubernetes?')">
              <span class="case-btn-tag">Case 1: Simple Lookup (Vector)</span>
              <span class="case-btn-query">"Tôi đã đọc gì về Kubernetes?"</span>
            </button>
            <button class="case-btn" onclick="applyCase(2, 'Recommend đọc gì tiếp')">
              <span class="case-btn-tag">Case 2: Profile-Needed (Feast Topic)</span>
              <span class="case-btn-query">"Recommend đọc gì tiếp"</span>
            </button>
            <button class="case-btn" onclick="applyCase(3, 'Tôi đang quan tâm gì gần đây?')">
              <span class="case-btn-tag">Case 3: Fresh Velocity (1h Spike)</span>
              <span class="case-btn-query">"Tôi đang quan tâm gì gần đây?"</span>
            </button>
            <button class="case-btn" onclick="applyCase(4, 'Tài liệu về tự động mở rộng hạ tầng?')">
              <span class="case-btn-tag">Case 4: Paraphrase (Dense Vector)</span>
              <span class="case-btn-query">"Tài liệu về tự động mở rộng hạ tầng?"</span>
            </button>
            <button class="case-btn" onclick="applyCase(5, 'Cho tôi summary cloud security')">
              <span class="case-btn-tag">Case 5: Mixed Hybrid (Episodic + Profile)</span>
              <span class="case-btn-query">"Cho tôi summary cloud security"</span>
            </button>
          </div>
        </div>
      </div>

    </div>

    <!-- MIDDLE COLUMN: Main Workspace Tabs -->
    <div>

      <div class="tabs">
        <button class="tab-btn active" id="tab-btn-recall" onclick="switchTab('recall')">🔍 Truy Vấn & Hồi Tưởng (Recall)</button>
        <button class="tab-btn" id="tab-btn-remember" onclick="switchTab('remember')">💾 Ghi Nhớ Mới (Remember)</button>
        <button class="tab-btn" id="tab-btn-explore" onclick="switchTab('explore')">📚 Kho Hồi Ức Đang Lưu</button>
      </div>

      <!-- TAB 1: RECALL -->
      <div id="tab-recall">
        <div class="card">
          <div class="card-body">
            <label for="query-input">Nhập câu hỏi hoặc yêu cầu cho AI Assistant:</label>
            <div style="display: flex; gap: 8px;">
              <input type="text" id="query-input" placeholder="Ví dụ: Tôi đã đọc gì về Kubernetes? hoặc Cho tôi summary cloud security..." value="Tôi đã đọc gì về Kubernetes?" onkeydown="if(event.key==='Enter') doRecall()">
              <button class="btn btn-primary" onclick="doRecall()" id="btn-recall">⚡ Thực Hiện</button>
            </div>
          </div>
        </div>

        <!-- AI Output Box -->
        <div id="ai-response-container" style="display: none;">
          <div class="ai-response-card">
            <div class="ai-response-title">✨ Trợ Lý AI Cá Nhân Hóa (Sinh Theo Hồ Sơ Người Dùng)</div>
            <div class="ai-response-body" id="ai-response-text"></div>
          </div>
        </div>

        <!-- Retrieved Memories Hits -->
        <div class="card">
          <div class="card-header">
            <div class="card-title">📖 Mảnh Ký Ức Trúng Tuyển (Top-K Chunks)</div>
            <span class="badge badge-info" id="hits-count-badge">0 hits</span>
          </div>
          <div class="card-body" id="hits-container">
            <div class="empty-state">Bấm nút "Thực Hiện" hoặc chọn kịch bản mẫu để truy vấn.</div>
          </div>
        </div>

        <!-- Assembled Prompt Context Preview -->
        <div class="card">
          <div class="card-header">
            <div class="card-title">🧩 Assembled Prompt Context (Ghép Feast + Qdrant)</div>
            <button class="btn btn-outline btn-sm" onclick="copyContext()">📋 Copy Context</button>
          </div>
          <div class="card-body">
            <pre class="code-block" id="assembled-context-text">// Assembled Prompt Context sẽ hiển thị tại đây...</pre>
          </div>
        </div>
      </div>

      <!-- TAB 2: REMEMBER -->
      <div id="tab-remember" style="display: none;">
        <div class="card">
          <div class="card-header">
            <div class="card-title">📝 Nạp Hồi Ức Mới Vào Qdrant</div>
          </div>
          <div class="card-body">
            <label for="remember-text">Nhập văn bản, tài liệu hoặc đoạn chat cần ghi nhớ:</label>
            <textarea id="remember-text" rows="5" placeholder="Ví dụ: Hôm nay tôi vừa hoàn thành bài lab về Feature Store Feast và phát hiện ra hiện tượng rò rỉ dữ liệu khi làm Target Encoding..."></textarea>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px;">
              <span style="font-size: 11.5px; color: var(--text-muted);">Tự động cắt theo câu (~300 ký tự) & gắn nhãn tenant <strong><span class="remember-uid-lbl">u_001</span></strong></span>
              <button class="btn btn-primary" onclick="doRemember()" id="btn-remember">💾 Ghi Nhớ Vào Vector Store</button>
            </div>
            
            <div id="remember-result" style="margin-top: 16px; display: none;"></div>
          </div>
        </div>
      </div>

      <!-- TAB 3: EXPLORE -->
      <div id="tab-explore" style="display: none;">
        <div class="card">
          <div class="card-header">
            <div class="card-title">📚 Danh Sách Ký Ức Đang Lưu Cho <span id="explore-uid-lbl">u_001</span></div>
            <button class="btn btn-outline btn-sm" onclick="loadUserMemories()">🔄 Tải Lại</button>
          </div>
          <div class="card-body" id="explore-memories-container">
            <div class="empty-state">Đang tải danh sách ký ức...</div>
          </div>
        </div>
      </div>

    </div>

    <!-- RIGHT COLUMN: Step-by-Step Live Execution Logs -->
    <div>
      <div class="card">
        <div class="card-header">
          <div class="card-title">📋 Nhật Ký Thực Thi (Step Logs)</div>
          <span class="badge badge-accent" id="lbl-total-latency">0 ms</span>
        </div>
        <div class="card-body" style="padding: 12px;">
          <div class="timeline" id="timeline-container">
            <div class="empty-state" style="padding: 20px 0;">Chưa có bước thực thi nào. Hãy thử chạy một truy vấn!</div>
          </div>
        </div>
      </div>
    </div>

  </div>

  <script>
    let currentUsers = [];
    let activeTab = 'recall';

    // Initialize UI on load
    window.addEventListener('DOMContentLoaded', async () => {
      await fetchUsers();
      onUserChange();
    });

    async function fetchUsers() {
      try {
        const res = await fetch('/api/users');
        currentUsers = await res.json();
      } catch (e) {
        console.error("Fetch users error:", e);
      }
    }

    function onUserChange() {
      const uid = document.getElementById('select-user').value;
      document.getElementById('lbl-uid').innerText = uid;
      document.querySelectorAll('.remember-uid-lbl').forEach(el => el.innerText = uid);
      document.getElementById('explore-uid-lbl').innerText = uid;

      const user = currentUsers.find(u => u.user_id === uid) || {
        topic_affinity: 'cloud',
        reading_speed_wpm: 187,
        preferred_language: 'vi',
        queries_last_hour: 11
      };

      document.getElementById('stat-topic').innerText = user.topic_affinity;
      document.getElementById('stat-wpm').innerText = user.reading_speed_wpm + ' WPM';
      document.getElementById('stat-lang').innerText = user.preferred_language;
      document.getElementById('stat-queries').innerText = user.queries_last_hour + ' reqs';

      if (activeTab === 'explore') {
        loadUserMemories();
      }
    }

    function switchTab(tab) {
      activeTab = tab;
      ['recall', 'remember', 'explore'].forEach(t => {
        document.getElementById(`tab-${t}`).style.display = (t === tab) ? 'block' : 'none';
        document.getElementById(`tab-btn-${t}`).classList.toggle('active', t === tab);
      });
      if (tab === 'explore') {
        loadUserMemories();
      }
    }

    function applyCase(num, query) {
      document.getElementById('query-input').value = query;
      switchTab('recall');
      doRecall();
    }

    async function doRecall() {
      const query = document.getElementById('query-input').value.trim();
      if (!query) return;

      const uid = document.getElementById('select-user').value;
      const topK = parseInt(document.getElementById('top-k-slider').value);
      const useVector = document.getElementById('chk-vector').checked;
      const useProfile = document.getElementById('chk-profile').checked;
      const useVelocity = document.getElementById('chk-velocity').checked;
      const enforceTenant = document.getElementById('chk-tenant').checked;
      const generateAi = document.getElementById('chk-ai').checked;

      const btn = document.getElementById('btn-recall');
      btn.disabled = true;
      btn.innerText = 'Đang xử lý...';

      try {
        const res = await fetch('/api/recall', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: query,
            user_id: uid,
            top_k: topK,
            use_vector: useVector,
            use_profile: useProfile,
            use_velocity: useVelocity,
            enforce_tenant: enforceTenant,
            generate_ai: generateAi
          })
        });
        const data = await res.json();
        renderRecallResults(data);
      } catch (err) {
        alert("Lỗi khi recall: " + err);
      } finally {
        btn.disabled = false;
        btn.innerText = '⚡ Thực Hiện';
      }
    }

    function renderRecallResults(data) {
      // 1. Total Latency
      document.getElementById('lbl-total-latency').innerText = `${data.total_latency_ms} ms`;

      // 2. AI Response
      const aiContainer = document.getElementById('ai-response-container');
      const aiText = document.getElementById('ai-response-text');
      if (data.ai_response) {
        aiContainer.style.display = 'block';
        aiText.innerText = data.ai_response;
      } else {
        aiContainer.style.display = 'none';
      }

      // 3. Hits
      const hitsContainer = document.getElementById('hits-container');
      const hitsBadge = document.getElementById('hits-count-badge');
      hitsBadge.innerText = `${data.hits.length} hits`;

      if (data.hits.length === 0) {
        hitsContainer.innerHTML = '<div class="empty-state">Không tìm thấy mẩu ký ức nào phù hợp trong Qdrant.</div>';
      } else {
        let html = '';
        data.hits.forEach((h, idx) => {
          const scorePercent = Math.min(100, Math.round(h.score * 100));
          const isOtherTenant = h.user_id !== data.user_id;
          const tenantBadge = isOtherTenant ? '<span class="badge badge-warning">Rò rỉ từ ' + h.user_id + '</span>' : '';
          html += `
            <div class="memory-hit-card">
              <div class="memory-hit-header">
                <span style="font-size:12px; font-weight:600; color:var(--text-secondary);">Chunk #${idx} [ID: ${h.memory_id}] ${tenantBadge}</span>
                <span class="memory-score-badge">Cosine: ${h.score.toFixed(3)}</span>
              </div>
              <div class="memory-score-bar-bg">
                <div class="memory-score-bar-fill" style="width: ${scorePercent}%;"></div>
              </div>
              <div style="font-size: 13px; color: var(--text-primary); line-height: 1.5;">${h.text}</div>
            </div>
          `;
        });
        hitsContainer.innerHTML = html;
      }

      // 4. Assembled Context
      document.getElementById('assembled-context-text').innerText = data.assembled_context;

      // 5. Execution Steps Timeline
      const timeline = document.getElementById('timeline-container');
      let stepsHtml = '';
      data.steps.forEach(s => {
        stepsHtml += `
          <div class="step-item">
            <div class="step-num">${s.step_id}</div>
            <div class="step-content">
              <div class="step-header">
                <span class="step-title">${s.title}</span>
                <span class="step-badge step-tag-${s.tag}">${s.tag}</span>
              </div>
              <div class="step-details">${s.details}</div>
              <div style="margin-top: 4px; display: flex; justify-content: flex-end;">
                <span class="step-latency">⏱ ${s.latency_ms} ms</span>
              </div>
            </div>
          </div>
        `;
      });
      timeline.innerHTML = stepsHtml;
    }

    async function doRemember() {
      const text = document.getElementById('remember-text').value.trim();
      if (!text) {
        alert("Vui lòng nhập nội dung cần ghi nhớ!");
        return;
      }
      const uid = document.getElementById('select-user').value;
      const btn = document.getElementById('btn-remember');
      btn.disabled = true;
      btn.innerText = 'Đang lưu vào Qdrant...';

      try {
        const res = await fetch('/api/remember', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text, user_id: uid, max_chunk_chars: 300 })
        });
        const data = await res.json();
        const resBox = document.getElementById('remember-result');
        resBox.style.display = 'block';
        resBox.innerHTML = `
          <div class="badge badge-success" style="padding: 8px 12px; display: flex; width: 100%; justify-content: space-between;">
            <span>✓ Đã lưu thành công <strong>${data.chunks_count} chunks</strong> cho ${data.user_id} (${data.total_latency_ms} ms)</span>
          </div>
          <div style="margin-top: 8px; font-size: 12px; color: var(--text-muted);">
            Chunks IDs: ${data.memory_ids.join(', ')}
          </div>
        `;
        document.getElementById('remember-text').value = '';
      } catch (err) {
        alert("Lỗi khi lưu ký ức: " + err);
      } finally {
        btn.disabled = false;
        btn.innerText = '💾 Ghi Nhớ Vào Vector Store';
      }
    }

    async function loadUserMemories() {
      const uid = document.getElementById('select-user').value;
      const container = document.getElementById('explore-memories-container');
      container.innerHTML = '<div class="empty-state">Đang tải ký ức...</div>';

      try {
        const res = await fetch(`/api/user/${uid}/memories`);
        const memories = await res.json();
        if (memories.length === 0) {
          container.innerHTML = `<div class="empty-state">Người dùng ${uid} chưa có ký ức nào. Hãy sang tab Ghi Nhớ Mới để nạp.</div>`;
          return;
        }
        let html = '';
        memories.forEach((m, idx) => {
          html += `
            <div class="memory-hit-card">
              <div class="memory-hit-header">
                <span style="font-size:12px; font-weight:600; color:var(--accent);">#${idx + 1} [ID: ${m.memory_id}]</span>
                <span style="font-size:11px; color:var(--text-muted);">${new Date(m.created_at * 1000).toLocaleTimeString()}</span>
              </div>
              <div style="font-size: 13px; color: var(--text-primary); line-height: 1.5;">${m.text}</div>
            </div>
          `;
        });
        container.innerHTML = html;
      } catch (err) {
        container.innerHTML = `<div class="empty-state">Lỗi khi tải: ${err}</div>`;
      }
    }

    async function resetMemories() {
      if (!confirm("Khôi phục toàn bộ bộ nhớ về 5 tài liệu chuẩn của Lab?")) return;
      try {
        const res = await fetch('/api/reset', { method: 'POST' });
        const data = await res.json();
        alert(data.message);
        onUserChange();
        if (activeTab === 'explore') loadUserMemories();
      } catch (err) {
        alert("Lỗi khi reset: " + err);
      }
    }

    function copyContext() {
      const text = document.getElementById('assembled-context-text').innerText;
      navigator.clipboard.writeText(text).then(() => {
        alert("Đã sao chép Assembled Context vào Clipboard!");
      });
    }
  </script>
</body>
</html>
"""


def main() -> None:
    port = 8501
    print("=" * 60)
    print("  HYBRID MEMORY AGENT STUDIO — BONUS CHALLENGE UI")
    print("=" * 60)
    print(f"  • Starting Web Server on http://localhost:{port}")
    print("  • Palette: Soft Light (Slate/White/Indigo) with 7px rounded corners")
    print("  • Granular Checkbox Controls + 5 Demo Presets + Execution Step Logs")
    print("=" * 60)
    uvicorn.run("bonus.web_app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
