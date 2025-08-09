from graphviz import Digraph

# Create the diagram
dot = Digraph(comment='LLM Document Processing System Data Flow - Updated Large Font', format='png')

# Global style: black background, white text, bigger font for PPT
dot.attr(bgcolor='black')
dot.attr('node', shape='box', style='filled', color='white', fillcolor='black',
         fontcolor='white', fontsize='16', fontname='Arial')
dot.attr('edge', color='white', fontsize='14', fontname='Arial')

# Frontend
dot.node('QI', 'User enters query / uploads documents')
dot.node('DR', 'Displays only the formal reply, which is the decision')

# Backend/API
dot.node('RC', 'Receive query and documents via REST / GraphQL')
dot.node('LP', 'Query Parser LLM (extract structured facts)')
dot.node('RQ', 'Generate query embedding')
dot.node('FRG', 'Formal Response Generator')

# Ingestion & Storage
dot.node('LD', 'Load and preprocess documents: PDF, Word, Email')
dot.node('CH', 'Chunk text and add metadata')
dot.node('EM', 'Generate embeddings (MiniLM-L6-v2)')
dot.node('VS', 'Vector DB (Milvus or pgvector)')

# Retrieval & Decision
dot.node('SR', 'Semantic Retrieval - find top relevant clauses')
dot.node('DE', 'Decision Engine LLM - compare facts and clauses')
dot.node('JO', 'Output structured JSON: decision, amount, justification, referenced_clauses')

# Audit
dot.node('AL', 'Log query, retrieved clauses, decision, and response')

# Connections
dot.edges([
    ('QI', 'RC'),
    ('RC', 'LP'),
    ('LP', 'RQ'),
    ('RQ', 'SR'),
    ('RC', 'LD'),
    ('LD', 'CH'),
    ('CH', 'EM'),
    ('EM', 'VS'),
    ('VS', 'SR'),
    ('SR', 'DE'),
    ('DE', 'JO'),
    ('JO', 'FRG'),
    ('FRG', 'DR'),
    ('JO', 'AL')
])

# Save & render PNG
dot.render('data_flow_ppt_updated_largefont', cleanup=True)
