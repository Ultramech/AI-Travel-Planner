# Legacy code (v1)

The original Streamlit-only implementation, kept for reference and diffing.
It is **not** imported by the current application and will not run as-is —
see the root `README.md` for the list of defects it contains.

Superseded by:

| Legacy file          | Replaced by                                               |
|----------------------|-----------------------------------------------------------|
| `app.py`             | `ui/streamlit_app.py` (thin API client) + `app/api/`      |
| `mainpy.py`          | `app/graph/` (state, nodes, routers, builder, checkpointer)|
| `trip_extractor.py`  | `app/providers/extractor.py` + `app/providers/geo.py`     |
| `airplane_data.py`   | `app/providers/flights.py`                                |
| `train_data.py`      | `app/providers/trains.py` + `app/scraping/driver.py`      |
| `bus_data.py`        | `app/providers/buses.py` + `app/scraping/driver.py`       |
| `cab_data.py`        | `app/providers/cabs.py`                                   |
| `hotels_data.py`     | `app/providers/hotels.py`                                 |
| `daywise_plan.py`    | `app/providers/itinerary.py`                              |

The notebooks (`main.ipynb`, `main3 - Copy.ipynb`) remain at the repo root.
