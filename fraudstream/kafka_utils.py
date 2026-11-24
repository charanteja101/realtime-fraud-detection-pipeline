from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


def ensure_topic(bootstrap: str, topic: str, partitions: int, wait_s: int = 90) -> None:
    """Create the topic if needed (waits for the broker to come up). Safe to call repeatedly."""
    from confluent_kafka import KafkaError, KafkaException
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap})
    deadline = time.time() + wait_s
    while True:
        try:
            md = admin.list_topics(timeout=5)
            break
        except KafkaException as exc:
            if time.time() > deadline:
                raise
            log.warning("broker not ready (%s), retrying", exc)
            time.sleep(2)
    if topic in md.topics and md.topics[topic].error is None:
        return
    futures = admin.create_topics([NewTopic(topic, num_partitions=partitions, replication_factor=1)])
    for f in futures.values():
        try:
            f.result()
        except KafkaException as exc:
            if exc.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise
