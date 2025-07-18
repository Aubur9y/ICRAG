import logging

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()


class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver:
            self.driver.close()

    def query(self, query, parameters=None, db=None):
        assert self.driver is not None, "Driver not initialised"
        session = None
        response = None

        try:
            session = (
                self.driver.session(database=db)
                if db is not None
                else self.driver.session()
            )
            response = list(session.run(query, parameters))
        except Exception as e:
            logging.error(f"Query failed: {e}")
        finally:
            if session is not None:
                session.close()
        return response


def create_connection():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = os.environ.get("NEO4J_PASSWORD")

    conn = Neo4jConnection(uri, user, password)
    return conn


def test_connection():
    conn = create_connection()
    try:
        result = conn.query("RETURN 'Hello Neo4j' as message")
        print(result)
    finally:
        conn.close()
