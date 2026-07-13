import asyncio
import requests
import json
import time
import inngest

base = "http://127.0.0.1:8288/v1"

# Step 1: Send a query event
async def send_event():
    client = inngest.Inngest(app_id="rag_app", is_production=False)
    result = await client.send(inngest.Event(name="rag/query_pdf_ai", data={"question": "what is this document about?", "top_k": 3}))
    return result[0]

print("=== Step 1: Sending query event ===")
event_id = asyncio.run(send_event())
print(f"Event ID: {event_id}")

# Step 2: Poll for runs
print("\n=== Step 2: Polling for runs ===")
for attempt in range(60):
    time.sleep(2)
    resp = requests.get(f"{base}/events/{event_id}/runs")
    runs = resp.json().get("data", [])
    print(f"\nAttempt {attempt+1}: Found {len(runs)} run(s)")
    for run in runs:
        run_id = run.get("run_id", "")
        status = run.get("status", "")
        output = run.get("output")
        ended = run.get("ended_at")
        print(f"  Run {run_id}: status={status}, ended_at={ended}, has_output={output is not None}, output={json.dumps(output)[:200] if output else 'None'}")
        
        # If completed, also try fetching run detail
        if status in ("Completed", "Succeeded", "Success", "Finished"):
            detail_resp = requests.get(f"{base}/runs/{run_id}")
            print(f"  Detail endpoint: status={detail_resp.status_code}")
            if detail_resp.status_code == 200:
                detail = detail_resp.json()
                print(f"  Detail data: {json.dumps(detail)[:500]}")
            else:
                print(f"  Detail error: {detail_resp.text[:200]}")
            print("\n=== DONE - Run completed ===")
            exit(0)
    
    # Check if all are failed
    if runs and all(r.get("status") in ("Failed", "Cancelled") for r in runs):
        print("\n=== ALL RUNS FAILED ===")
        exit(1)

print("\n=== TIMED OUT ===")
