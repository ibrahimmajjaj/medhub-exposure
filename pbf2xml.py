import sys, osmium
from xml.sax.saxutils import quoteattr
src=sys.argv[1]
ROAD={"motorway","motorway_link","trunk","trunk_link","primary","primary_link","secondary","secondary_link"}
nodes={}; roads=[]; rails=[]
for o in osmium.FileProcessor(src):
    if o.is_node():
        nodes[o.id]=(o.location.lat,o.location.lon)
    elif o.is_way():
        tags={t.k:t.v for t in o.tags}
        refs=[n.ref for n in o.nodes]
        if tags.get("highway") in ROAD: roads.append((o.id,refs,tags))
        if tags.get("railway")=="rail": rails.append((o.id,refs,tags))
def write(fn, ways):
    used=set(r for _,refs,_ in ways for r in refs)
    with open(fn,"w",encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6">\n')
        for n in used:
            if n in nodes: f.write(f'<node id="{n}" version="1" lat="{nodes[n][0]}" lon="{nodes[n][1]}"/>\n')
        for wid,refs,tags in ways:
            refs=[r for r in refs if r in nodes]
            if len(refs)<2: continue
            f.write(f'<way id="{wid}" version="1">'+"".join(f'<nd ref="{r}"/>' for r in refs)+"".join(f'<tag k={quoteattr(k)} v={quoteattr(v)}/>' for k,v in tags.items())+"</way>\n")
        f.write("</osm>\n")
write("roads.osm",roads); write("rail.osm",rails)
print(len(nodes),"nodes",len(roads),"road ways",len(rails),"rail ways")
