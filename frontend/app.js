const $=id=>document.getElementById(id);
const uid=()=>$('uid').value.trim()||'chaithanya';
let cy=null;

const SAMPLES=[
 "Who knows about GraphRAG?",
 "Trace lineage of Graph RAG Survey",
 "What contradicts the claim that graph retrieval beats vector-only?",
 "find papers on graph retrieval augmented generation",
 "most influential paper on graph retrieval",
 "What did I ask before?",
];
SAMPLES.forEach(s=>{const b=document.createElement('button');b.className='sample';b.textContent='▸ '+s;b.onclick=()=>{$('q').value=s;ask();};$('samples').appendChild(b);});

function bubble(who,md,badge){
 let d=document.createElement('div');
 d.className=who==='u'?'msg-u':'msg-a';
 d.innerHTML=(badge?`<span class="toolbadge">◈ ${badge}</span><br>`:'')+marked.parse(md);
 $('chat').appendChild(d);$('chat').scrollTop=1e6;
}

function draw(nodes,edges,highlight=[]){
 const els=[
  ...nodes.map(n=>({data:{id:n.id,label:n.label,type:n.type},
    classes:(highlight.includes(n.id)?'hl ':'')+n.type})),
  ...edges.map((e,i)=>({data:{id:'e'+i,source:e.from,target:e.to,label:e.label}}))];
 if(cy)cy.destroy();
 cy=cytoscape({container:$('cy'),elements:els,
  style:[
   {selector:'node',style:{'label':'data(label)','font-size':9,'color':'#e2e8f0','text-wrap':'wrap','text-max-width':90,'background-color':'#334155','border-width':2,'border-color':'#64748b','width':34,'height':34,'text-valign':'bottom','text-margin-y':6}},
   {selector:'.Paper',style:{'background-color':'#8b5cf6','border-color':'#c4b5fd'}},
   {selector:'.Claim',style:{'background-color':'#f59e0b','border-color':'#fde68a'}},
   {selector:'.Method',style:{'background-color':'#22d3ee','border-color':'#a5f3fc'}},
   {selector:'.Person',style:{'background-color':'#34d399','border-color':'#a7f3d0'}},
   {selector:'.Project',style:{'background-color':'#f472b6','border-color':'#fbcfe8'}},
   {selector:'.Topic',style:{'background-color':'#38bdf8','border-color':'#bae6fd'}},
   {selector:'.Venue',style:{'background-color':'#64748b','border-color':'#cbd5e1'}},
   {selector:'.Dataset',style:{'background-color':'#a3e635','border-color':'#d9f99d'}},
   {selector:'.Episode',style:{'background-color':'#fb7185','border-color':'#fecdd3','shape':'diamond'}},
   {selector:'.Task',style:{'background-color':'#fbbf24','border-color':'#fef3c7','shape':'hexagon'}},
   {selector:'.Procedure',style:{'background-color':'#c084fc','border-color':'#e9d5ff','shape':'triangle'}},
   {selector:'.hl',style:{'border-color':'#facc15','border-width':4,'background-color':'#a16207'}},
   {selector:'edge',style:{'label':'data(label)','font-size':8,'color':'#94a3b8','line-color':'#475569','target-arrow-shape':'triangle','target-arrow-color':'#475569','curve-style':'bezier'}}],
  layout:{name:'cose',padding:30}});
}

async function api(p,o={}){
 const r=await fetch(p,{...o,headers:{'Content-Type':'application/json'}});
 if(!r.ok){let d=p;try{d=JSON.stringify(await r.json()).slice(0,160);}catch{}throw new Error(d);}
 return r.json();
}

async function refreshGraph(){
 const g=await api('/api/graph?user_id='+encodeURIComponent(uid()));
 draw(g.nodes,g.edges);
 $('stP').textContent=g.nodes.length+((g.hidden?' (+'+g.hidden+' hidden)':''));
 $('stE').textContent=g.edges.length;
}

function engineNote(r){
 const e=r.engines||{};
 return `graph: ${r.mode} · PageRank: ${e.pagerank||'—'} · communities: ${e.community||'—'}`;
}

async function ask(){
 const q=$('q').value.trim();if(!q)return;
 bubble('u',q);$('q').value='';
 bubble('a','_traversing graph…_');
 try{
  const r=await api('/api/ask',{method:'POST',body:JSON.stringify({user_id:uid(),question:q})});
  $('chat').lastChild.remove();
  const s=r.stats||{};
  bubble('a',r.text+`\n\n_router: ${r.router||'—'} · llm: ${r.llm||'template'} · ${engineNote(r)} · memories: ${s.memories||0} · episodes: ${s.episodes||0} · procedures: ${s.procedures||0}_`,
   `${r.tool} · FalkorDB traversal`);
  $('cypher').textContent=r.cypher+(r.cypher_params?`\n-- params: ${JSON.stringify(r.cypher_params)}`:'');
  draw(r.nodes,r.edges,r.highlight||[]);
  $('stP').textContent=(r.total_nodes||r.nodes.length)+(r.hidden?` (${r.nodes.length} shown, +${r.hidden} hidden)`:` (${r.nodes.length} shown)`);
  $('stE').textContent=r.stats.edges;
  $('stM').textContent=r.stats.memories;
 }catch(e){$('chat').lastChild.remove();bubble('a','**Error:** '+e.message+'\n\nIs the backend running? `python -m uvicorn app.main:app --port 8000`');}
}
$('btnAsk').onclick=ask;
$('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
$('btnInit').onclick=async()=>{try{
  const r=await api('/api/init',{method:'POST',body:JSON.stringify({user_id:uid()})});
  $('modeBadge').textContent='mode: '+r.mode;
  bubble('a','Graph `user_'+uid()+'` ready ('+r.mode+' mode) — seed citation chain `FalkorDB Memory → Graph RAG Survey → Vector-Only Limits`, claims with `CONTRADICTS`, expert edges `WROTE/ABOUT` all MERGED. Ask a sample ←');
  refreshGraph();
}catch(e){bubble('a','init failed: '+e.message);}};
$('btnMem').onclick=async()=>{const m=await api('/api/memory?user_id='+encodeURIComponent(uid()));
 bubble('a','**Persistent memory from graph (`User)-[:REMEMBERS]->(Episode):**\n'+(m.memories.map(x=>'- '+x).join('\n')||'- (empty — ask something first)'));};
$('btnGraph').onclick=()=>refreshGraph().catch(e=>bubble('a','graph refresh failed: '+e.message));
$('btnIngest').onclick=async()=>{bubble('a','_fetching Semantic Scholar + OpenAlex + arXiv: '+$('topic').value+'…_');try{
  const r=await api('/api/ingest',{method:'POST',body:JSON.stringify({user_id:uid(),topic:$('topic').value,max_results:6})});
  $('chat').lastChild.remove();
  bubble('a','**Ingested '+r.ingested+' papers** into your tenant graph: '+r.papers.map(p=>p.title.slice(0,40)).join(' · '),'/api/ingest · live sources → graph');
  refreshGraph();
}catch(e){$('chat').lastChild.remove();bubble('a','ingest failed: '+e.message);}};

$('btnAgents').onclick=async()=>{
 bubble('a','_Researcher → Critic → Librarian handoff (state lives in Task nodes)…_');
 try{
  const r=await api('/api/agents/run',{method:'POST',body:JSON.stringify({user_id:uid(),topic:$('topic').value})});
  $('chat').lastChild.remove();
  bubble('a','**Multi-agent run**\n- Researcher: '+r.researcher.papers+' papers ingested, '+r.researcher.handoffs+' Tasks created\n- Critic: '+r.critic.reviewed+' reviewed, '+r.critic.flagged+' flagged — decided by *'+r.critic.decided_by+'*, '+r.critic.experts_assigned+' expert(s) assigned\n- Librarian: '+r.librarian.filed+' filed, '+r.librarian.topics+' topic(s)\n\n_Handoffs are queryable: `MATCH (u:User)-[:OWNS]->(t:Task)-[:ABOUT]->(p:Paper) RETURN t.status`_',
   'agents pipeline · shared graph state');
  refreshGraph();
 }catch(e){$('chat').lastChild.remove();bubble('a','agents run failed: '+e.message);}
};

$('btnAlgo').onclick=async()=>{
 try{
  const r=await api('/api/algorithms?user_id='+encodeURIComponent(uid()));
  const top=(r.pagerank||[]).slice(0,5).map((p,i)=>`${i+1}. **${(p.title||p.id).slice(0,52)}** — \`${(p.score||0).toFixed(5)}\``).join('\n')||'_(no citation edges yet)_';
  bubble('a','**Graph algorithms as first-class tools**\n_PageRank over `CITES` — most influential papers:_\n'+top+
   `\n\n_communities: ${(r.communities||[]).length} · citation edges: ${r.citation_edges} · engines: ${JSON.stringify(r.engines)}_`,
   'algo.pagerank + louvain + components');
 }catch(e){bubble('a','algorithms failed: '+e.message);}
};

(async()=>{try{const h=await api('/api/health');$('modeBadge').textContent='mode: '+h.graph_mode+(h.mistral?' · mistral':' · template');
  }catch{$('modeBadge').textContent='mode: offline — start backend';}
 bubble('a','Welcome to **TraceMind**. I reason over a FalkorDB graph — citations, claims, people, decisions — and I **remember you across sessions**. Every answer shows its Cypher + routing. Pick a sample ← or run the agents pipeline.');
 refreshGraph().catch(()=>{});
})();
