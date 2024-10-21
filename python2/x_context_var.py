import asyncio
from contextvars import ContextVar

request_id_var = ContextVar('request_id', default=None)

async def handle_request(request_id):
    request_id_var.set(request_id)
    
    data = await process_data()
    await log_request(data)

async def process_data():
    request_id = request_id_var.get()
    data = int(request_id) * 2
    await asyncio.sleep(1)
    return data

async def log_request(data):
    request_id = request_id_var.get()
    print(f"Logging request {request_id} -> {data}")
    await asyncio.sleep(1)

async def main():
    await asyncio.gather(
        handle_request('123'),
        handle_request('456'),
        handle_request('789')
    )

asyncio.run(main())