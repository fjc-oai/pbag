import asyncio
from fastapi import FastAPI
from datetime import datetime
import uvicorn


def service_hanlder() -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"message": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    def _heavy_work():
        import time

        time.sleep(7)
        return 7

    @app.get("/heavy_work")
    async def heavy_work():
        resp = await asyncio.to_thread(_heavy_work)
        return {"message": resp}

    return app


def service():
    app = service_hanlder()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    service()
