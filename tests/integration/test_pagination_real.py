"""
List endpoints against REAL PostgreSQL.

Before this, GET /conversations, GET /conversations/{id} and GET /documents loaded
every row a tenant owned into memory and returned it in one response -- a cost that
grows without bound with the tenant's history. They are now paginated with a hard
ceiling. The response body is still a JSON array (existing clients keep working);
the total comes back in the X-Total-Count header.
"""

import uuid

import pytest

from integration_helpers import db_exec, register_tenant

pytestmark = pytest.mark.integration


def seed_calls(business_id, n, prefix="sess"):
    db_exec(
        "INSERT INTO calls (id, business_id, channel, caller_ref, status, started_at) "
        "SELECT gen_random_uuid(), :b, 'browser', :p || '-' || g, 'completed', now() - (g || ' seconds')::interval "
        "FROM generate_series(1, :n) g", b=str(business_id), p=prefix, n=n)


def seed_documents(business_id, n):
    db_exec(
        "INSERT INTO documents (id, business_id, filename, file_type, status, uploaded_at) "
        "SELECT gen_random_uuid(), :b, 'doc-' || g || '.txt', 'text', 'ready', now() - (g || ' seconds')::interval "
        "FROM generate_series(1, :n) g", b=str(business_id), n=n)


def all_pages(client, tenant, path, page_size):
    """Walk every page; returns (ids in order, X-Total-Count)."""
    ids, offset, total = [], 0, None
    while True:
        r = client.get(path, headers=tenant.auth, params={"limit": page_size, "offset": offset})
        assert r.status_code == 200, r.text
        total = int(r.headers["x-total-count"])
        page = r.json()
        ids.extend(item["id"] for item in page)
        if len(page) < page_size:
            return ids, total
        offset += page_size


# --------------------------------------------------------------- conversations

def test_conversations_default_page_is_bounded_and_reports_the_total(client):
    tenant = register_tenant(client, "Alpha")
    seed_calls(tenant.business_id, 130)

    r = client.get("/api/v1/conversations", headers=tenant.auth)

    assert r.status_code == 200 and isinstance(r.json(), list)
    assert len(r.json()) == 100                       # the default page, not all 130
    assert r.headers["x-total-count"] == "130"


def test_walking_the_pages_returns_every_conversation_exactly_once_newest_first(client):
    tenant = register_tenant(client, "Alpha")
    seed_calls(tenant.business_id, 53)

    ids, total = all_pages(client, tenant, "/api/v1/conversations", page_size=10)

    assert total == 53 and len(ids) == 53 and len(set(ids)) == 53   # no gaps, no repeats across pages
    first_page = client.get("/api/v1/conversations", headers=tenant.auth, params={"limit": 3}).json()
    assert [c["started_at"] for c in first_page] == sorted((c["started_at"] for c in first_page), reverse=True)


def test_pagination_never_counts_or_returns_another_tenants_rows(client):
    a, b = register_tenant(client, "Alpha"), register_tenant(client, "Bravo")
    seed_calls(a.business_id, 7, prefix="a")
    seed_calls(b.business_id, 4, prefix="b")

    ids, total = all_pages(client, a, "/api/v1/conversations", page_size=3)

    assert total == 7 and len(ids) == 7
    b_ids, b_total = all_pages(client, b, "/api/v1/conversations", page_size=3)
    assert b_total == 4 and not set(ids) & set(b_ids)


@pytest.mark.parametrize("params", [
    {"limit": 0}, {"limit": -1}, {"limit": 501}, {"offset": -1}, {"limit": "abc"},
])
def test_out_of_range_paging_parameters_are_rejected_not_clamped_silently(client, params):
    tenant = register_tenant(client, "Alpha")
    for path in ("/api/v1/conversations", "/api/v1/documents"):
        assert client.get(path, headers=tenant.auth, params=params).status_code == 422, (path, params)


def test_an_offset_past_the_end_is_an_empty_page_not_an_error(client):
    tenant = register_tenant(client, "Alpha")
    seed_calls(tenant.business_id, 3)
    r = client.get("/api/v1/conversations", headers=tenant.auth, params={"offset": 50})
    assert r.status_code == 200 and r.json() == [] and r.headers["x-total-count"] == "3"


# ------------------------------------------------------------------- documents

def test_documents_are_paginated_with_a_total(client):
    tenant = register_tenant(client, "Alpha")
    seed_documents(tenant.business_id, 120)

    default = client.get("/api/v1/documents", headers=tenant.auth)
    assert len(default.json()) == 100 and default.headers["x-total-count"] == "120"

    ids, total = all_pages(client, tenant, "/api/v1/documents", page_size=25)
    assert total == 120 and len(set(ids)) == 120


# ------------------------------------------------------- one conversation's messages

def test_a_conversation_returns_a_bounded_slice_of_its_messages_with_the_total(client):
    tenant = register_tenant(client, "Alpha")
    call_id = uuid.uuid4()
    db_exec("INSERT INTO calls (id, business_id, channel, caller_ref, status, started_at) "
            "VALUES (:c, :b, 'browser', 'big', 'completed', now())", c=str(call_id), b=str(tenant.business_id))
    db_exec("INSERT INTO messages (id, call_id, role, content, created_at) "
            "SELECT gen_random_uuid(), :c, CASE WHEN g % 2 = 1 THEN 'customer' ELSE 'agent' END, 'm' || g, "
            "now() + (g || ' milliseconds')::interval FROM generate_series(1, 60) g", c=str(call_id))

    page = client.get(f"/api/v1/conversations/{call_id}", headers=tenant.auth, params={"limit": 25}).json()
    assert [m["content"] for m in page["messages"]] == [f"m{i}" for i in range(1, 26)]   # oldest first
    assert page["message_count"] == 60

    rest = client.get(f"/api/v1/conversations/{call_id}", headers=tenant.auth, params={"limit": 25, "offset": 50}).json()
    assert [m["content"] for m in rest["messages"]] == [f"m{i}" for i in range(51, 61)]
