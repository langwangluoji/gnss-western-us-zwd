from pathlib import Path
import urllib.request,hashlib
out=Path(__file__).resolve().parent
url='https://www.cs.utexas.edu/~ecprice/papers/ambientgan.pdf'
try:
 with urllib.request.urlopen(url,timeout=25) as r:data=r.read()
 assert data[:4]==b'%PDF'
 p=out/'ambientgan_author_copy.pdf';p.write_bytes(data)
 import fitz
 doc=fitz.open(p)
 selected=[]
 for i in range(min(5,len(doc))):
  t=doc[i].get_text()
  for term in ['identif','unique','measurements']:
   j=t.lower().find(term)
   if j>=0:selected.append({'page':i+1,'term':term,'excerpt':t[max(0,j-100):j+600]})
 import json
 (out/'ambientgan_primary_excerpts.json').write_text(json.dumps({'source':url,'sha256':hashlib.sha256(data).hexdigest(),'pages':len(doc),'excerpts':selected},indent=2),encoding='utf8')
 print(json.dumps(selected[:5]))
except Exception as e: print(type(e).__name__,str(e))
