import os

import statsd

metrics = statsd.StatsClient(os.getenv("STATSD_HOST"), 8125)
