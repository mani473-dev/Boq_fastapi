from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from main import executeSql

mcp = FastMCP(
    "SQL Agent Server",
    stateless_http=True,
    json_response=True, 
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    ),
)

@mcp.tool()
def getOutput(payload: dict):
    
    result = executeSql(payload)
    return result

app = mcp.streamable_http_app()