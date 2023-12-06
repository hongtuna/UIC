from Polar import polar
from bleak.uuids import uuid16_dict
import os, asyncio


if __name__ == "__main__":
    polar = polar()
    os.environ["PYTHONASYNCIODEBUG"] = str(1)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(polar.main())
