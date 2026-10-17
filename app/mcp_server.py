"""MCP Server exposing graph tools for LLM tool calling."""
from typing import Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.graph import (
    query_contradictions, run_pagerank, shortest_path_between,
    community_detection, get_interests, get_procedures, influential_in_topic,
    create_task, get_tasks, update_task_status, sanitize_user_id
)
from app.llm import TOOL_SPECS

app = FastAPI(title="TraceMind MCP Server")

class MCPTool(BaseModel):
    name: str
    description: str
    inputSchema: dict

TOOLS = [
    MCPTool(
        name="find_contradictions",
        description="Find contradictory claims between papers in the graph. Returns claim pairs with evidence.",
        inputSchema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}, "limit": {"type": "integer", "default": 10}},
            "required": ["user_id"]
        }
    ),
    MCPTool(
        name="pagerank",
        description="Run PageRank over CITES edges to find most influential papers in the user's graph.",
        inputSchema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}, "top_k": {"type": "integer", "default": 20}},
            "required": ["user_id"]
        }
    ),
    MCPTool(
        name="shortest_path",
        description="Find shortest citation path between two papers.",
        inputSchema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}, "paper_a": {"type": "string"}, "paper_b": {"type": "string"}, "max_depth": {"type": "integer", "default": 6}},
            "required": ["user_id", "paper_a", "paper_b"]
        }
    ),
    MCPTool(
        name="community_detection",
        description="Run community detection (Louvain) on CITES graph to cluster papers by topic.",
        inputSchema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}, "min_size": {"type": "integer", "default": 3}},
            "required": ["user_id"]
        }
    ),
    MCPTool(
        name="who_knows",
        description="Find experts (authors) for a given method/topic.",
        inputSchema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}, "method": {"type": "string"}},
            "required": ["user_id", "method"]
        }
    ),
    MCPTool(
        name="get_user_interests",
        description="Get user's semantic interests (topics with weights).",
        inputSchema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}, "top_k": {"type": "integer", "default": 10}},
            "required": ["user_id"]
        }
    ),
    MCPTool(
        name="get_procedures",
        description="Retrieve stored query plans (procedural memory) for similar questions.",
        inputSchema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}, "intent": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
            "required": ["user_id"]
        }
    ),
    MCPTool(
        name="create_handoff_task",
        description="Create a task for agent handoff (Researcher -> Critic -> Librarian).",
        inputSchema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}, "from_agent": {"type": "string"}, "to_agent": {"type": "string"}, "paper_id": {"type": "string"}, "payload": {"type": "object"}},
            "required": ["user_id", "from_agent", "to_agent", "paper_id"]
        }
    ),
    MCPTool(
        name="get_agent_tasks",
        description="Get tasks assigned to an agent.",
        inputSchema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}, "agent": {"type": "string"}, "status": {"type": "string"}},
            "required": ["user_id", "agent"]
        }
    ),
    MCPTool(
        name="influential_papers",
        description="Most influential papers in a topic: PageRank over CITES, scoped to the topic's papers.",
        inputSchema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}, "topic": {"type": "string"}, "top_k": {"type": "integer", "default": 5}},
            "required": ["user_id", "topic"]
        }
    ),
    MCPTool(
        name="run_agents",
        description="Run the Researcher -> Critic -> Librarian handoff pipeline over a topic.",
        inputSchema={
            "type": "object",
            "properties": {"user_id": {"type": "string"}, "topic": {"type": "string"}},
            "required": ["user_id", "topic"]
        }
    ),
]

@app.get("/mcp/tools")
def list_tools():
    return {"tools": [t.model_dump() for t in TOOLS],
            "router_descriptions": TOOL_SPECS}

@app.post("/mcp/call")
async def call_tool(request: Request):
    body = await request.json()
    name = body.get("name")
    args = body.get("arguments", {})
    user_id = args.get("user_id")

    if not user_id:
        return JSONResponse(status_code=400, content={"error": "user_id required"})

    try:
        user_id = sanitize_user_id(user_id)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        if name == "find_contradictions":
            result = query_contradictions(user_id, args.get("limit", 10))
        elif name == "pagerank":
            result = run_pagerank(user_id, top_k=args.get("top_k", 20))
        elif name == "shortest_path":
            result = shortest_path_between(user_id, args["paper_a"], args["paper_b"], args.get("max_depth", 6))
        elif name == "community_detection":
            result = community_detection(user_id, args.get("min_size", 3))
        elif name == "who_knows":
            from app.graph import find_experts
            result = find_experts(user_id, args["method"], limit=args.get("limit", 5))
        elif name == "get_user_interests":
            result = get_interests(user_id, args.get("top_k", 10))
        elif name == "get_procedures":
            result = get_procedures(user_id, args.get("intent"), args.get("limit", 5))
        elif name == "create_handoff_task":
            create_task(user_id, args["from_agent"], args["to_agent"], args["paper_id"], payload=args.get("payload"))
            result = {"ok": True}
        elif name == "get_agent_tasks":
            result = get_tasks(user_id, status=args.get("status"), to_agent=args["agent"])
        elif name == "influential_papers":
            result = influential_in_topic(user_id, args["topic"], top_k=args.get("top_k", 5))
        elif name == "run_agents":
            from app.agents import run_pipeline
            result = run_pipeline(user_id, args["topic"])
        else:
            return JSONResponse(status_code=404, content={"error": f"Unknown tool: {name}"})

        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)