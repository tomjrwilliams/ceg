
# TODO: register a plugin that on update of a node (say a table node)

# pushes the table data as json with an ID

# then app side we built up a big if statement over IDs to components

# that updates the relevant component, on receipt from the socket

# so we can then spawn the consumer (gui) in one process

# and then push data into it from the graph in another

# conversely, presumably can (for instance), have a queue object between the two

# so we can pass events from the gui back into the graph (eg. a loop node)

# https://github.com/ash2shukla/streamlit-stream/blob/master/producer/main.py

# def example_producer():
#     from fastapi import FastAPI, WebSocket
#     from random import choice, randint
#     import asyncio


#     app = FastAPI()

#     CHANNELS = ["A", "B", "C"]

#     @app.websocket("/sample")
#     async def websocket_endpoint(websocket: WebSocket):
#         await websocket.accept()
#         while True:
#             await websocket.send_json({
#                 "channel": choice(CHANNELS),
#                 "data": randint(1, 10)
#                 }
#             )
#             await asyncio.sleep(0.5)

# def example_consumer():
#     import aiohttp
#     from collections import deque, defaultdict
#     from functools import partial
#     from os import getenv
#     if getenv("IS_DOCKERIZED"):
#         WS_CONN = "ws://wsserver/sample"
#     else:
#         WS_CONN = "ws://localhost:8000/sample"
#     async def consumer(graphs, selected_channels, window_size, status):
#         windows = defaultdict(partial(deque, [0]*window_size, maxlen=window_size))

#         async with aiohttp.ClientSession(trust_env = True) as session:
#             status.subheader(f"Connecting to {WS_CONN}")
#             async with session.ws_connect(WS_CONN) as websocket:
#                 status.subheader(f"Connected to: {WS_CONN}")
#                 async for message in websocket:
#                     data = message.json()

#                     windows[data["channel"]].append(data["data"])

#                     for channel, graph in graphs.items():
#                         channel_data = {channel: windows[channel]}
#                         if channel == "A":
#                             graph.line_chart(channel_data)
#                         elif channel == "B":
#                             graph.area_chart(channel_data)
#                         elif channel == "C":
#                             graph.bar_chart(channel_data)

# def example_app():
#     import asyncio
#     import streamlit as st
#     from utils import consumer


#     st.set_page_config(page_title="stream", layout="wide")

#     status = st.empty()
#     connect = st.checkbox("Connect to WS Server")

#     selected_channels = st.multiselect("Select Channels", ["A", "B", "C"], default=["A"])

#     columns = [col.empty() for col in st.columns(len(selected_channels))]


#     window_size = st.number_input("Window Size", min_value=10, max_value=100)

#     if connect:
#         asyncio.run(consumer(dict(zip(selected_channels, columns)), selected_channels, window_size, status))
#     else:
#         status.subheader(f"Disconnected.")