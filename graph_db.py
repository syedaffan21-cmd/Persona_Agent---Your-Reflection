import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

class GraphDatabaseManager:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD")
        if not self.uri or not self.password:
            print("NEO4J_URI / NEO4J_PASSWORD not set in environment - graph writes will be disabled.")
        self.driver = None
        self.connect()

    def connect(self):
        try:
            uri = self.uri
            if uri and uri.startswith("neo4j+s://"):
                uri = uri.replace("neo4j+s://", "neo4j+ssc://")
                
            self.driver = GraphDatabase.driver(uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            print("Graph Manager successfully connected to Neo4j Aura!")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def add_fact(self, entity: str, relation: str, target: str):
        if not self.driver:
            return
        query = """
        MERGE (e:Entity {name: $entity})
        MERGE (t:Entity {name: $target})
        MERGE (e)-[r:%s]->(t)
        """ % relation
        try:
            with self.driver.session() as session:
                session.run(query, entity=entity.strip(), target=target.strip())
        except Exception as e:
            print(f"Error adding fact: {e}")

    def get_related_facts(self, entity_name: str) -> list[str]:
        if not self.driver:
            return []
        
        facts = []
        try:
            with self.driver.session() as session:
                query = """
                MATCH (e:Entity {name: $entity_name})-[r]->(t)
                RETURN e.name AS subject, type(r) AS relation, t.name AS target
                LIMIT 10
                """
                result = session.run(query, entity_name=entity_name)
                for record in result:
                    facts.append(f"{record['subject']} {record['relation']} {record['target']}")
        except Exception as e:
            print(f"Error fetching graph facts: {e}")
        return facts

graph_db = GraphDatabaseManager()