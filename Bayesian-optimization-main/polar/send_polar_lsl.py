
# general imports
import os, asyncio
from bleak.uuids import uuid16_dict

# change path to the main directory
# os.chdir('../')

print(os.getcwd())

# Polar imports
from utils.Polar import polar


if __name__ == "__main__":
    polar = polar(address='E1:26:4D:8F:18:3B')
    os.environ["PYTHONASYNCIODEBUG"] = str(1)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(polar.main())
