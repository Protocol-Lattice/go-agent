const CodeModeUI = (() => {
  const state = { nodes: [], active: null, sequence: 0, sawEvent: false };

  function ensurePanel() {
    let panel = document.getElementById('codemodeWorkflow');
    if (panel) return panel;
    panel = document.createElement('section');
    panel.id = 'codemodeWorkflow'; panel.className = 'codemode-workflow hidden';
    panel.innerHTML = `<div class="codemode-header"><div><span class="codemode-eyebrow">CODEMODE</span><strong>Tool workflow</strong></div><span id="codemodeStatus" class="codemode-status">idle</span></div><div id="codemodeGraph" class="codemode-graph"></div>`;
    document.getElementById('messages')?.prepend(panel); return panel;
  }
  function setStatus(text, kind='') { const el=document.getElementById('codemodeStatus'); if(!el)return; el.textContent=text; el.className=`codemode-status ${kind}`.trim(); }
  function render() {
    const panel=ensurePanel(); panel.classList.remove('hidden'); const graph=document.getElementById('codemodeGraph'); if(!graph)return;
    graph.replaceChildren(...state.nodes.map((node,index)=>{ const wrapper=document.createElement('div'); wrapper.className='codemode-node-wrap'; if(index>0){const edge=document.createElement('div');edge.className='codemode-edge';edge.innerHTML='<span>↓</span>';wrapper.append(edge)}
      const card=document.createElement('article'); card.className=`codemode-node ${node.status||''}`.trim();
      const head=document.createElement('div');head.className='codemode-node-head'; const marker=document.createElement('span');marker.className='codemode-node-marker';marker.textContent=node.status==='done'?'✓':node.status==='error'?'!':node.status==='running'?'●':String(index+1);
      const title=document.createElement('div');title.className='codemode-node-title';const strong=document.createElement('strong');strong.textContent=node.title;const meta=document.createElement('span');meta.textContent=node.tool;title.append(strong,meta);const badge=document.createElement('span');badge.className='codemode-node-badge';badge.textContent=node.status;head.append(marker,title,badge);
      const details=document.createElement('div');details.className='codemode-node-details'; if(node.args){const args=document.createElement('pre');args.className='codemode-code';args.textContent=JSON.stringify(node.args,null,2);details.append(args)} if(node.code){const code=document.createElement('pre');code.className='codemode-code';code.textContent=node.code;details.append(code)} if(node.output){const output=document.createElement('pre');output.className='codemode-output';output.textContent=node.output;details.append(output)} if(node.duration){const duration=document.createElement('span');duration.className='codemode-duration';duration.textContent=node.duration;details.append(duration)} card.append(head,details);wrapper.append(card);return wrapper; }));
  }
  function addNode(title,tool,code='') { const node={id:`codemode-${++state.sequence}`,title,tool,code,status:'running',startedAt:performance.now()};state.nodes.push(node);state.active=node;setStatus(`${state.nodes.length} node${state.nodes.length===1?'':'s'} · running`,'running');render();return node; }
  function finishNode(output='') { if(!state.active)return;state.active.status='done';state.active.output=output;state.active.duration=`${Math.max(1,Math.round(performance.now()-state.active.startedAt))} ms`;state.active=null;setStatus(`${state.nodes.length} node${state.nodes.length===1?'':'s'} · complete`,'done');render(); }
  function fail(message){if(state.active){state.active.status='error';state.active.output=message;state.active.duration=`${Math.max(1,Math.round(performance.now()-state.active.startedAt))} ms`;state.active=null}setStatus('workflow failed','error');render()}
  function reset(){state.nodes.length=0;state.active=null;state.sawEvent=false;ensurePanel().classList.add('hidden');setStatus('idle');}
  function extractCode(text){const value=String(text||'');const fenced=value.match(/```(?:go)?\s*([\s\S]*?)```/i);if(fenced)return fenced[1].trim();const code=value.match(/\b(?:package|func|fmt\.|client\.|tools\.).*[\s\S]*/i);return code?code[0].trim():''}

  function onToolEvent(event){
    if(!event?.type||!event.tool)return;
    state.sawEvent=true;
    if(event.type==='codemode_tool_start') { addNode(`Execute ${event.tool}`,event.tool); if(event.args&&state.active){state.active.args=event.args;render()} return; }
    const node=state.nodes.find(item=>item.tool===event.tool&&item.status==='running')||state.active;
    if(!node)return;
    node.duration=`${Math.max(1,Number(event.duration_ms||0))} ms`;
    if(event.type==='codemode_tool_error'){node.status='error';node.output=event.error||'Tool execution failed';if(state.active===node)state.active=null;setStatus('workflow failed','error');render();return}
    if(event.type==='codemode_tool_result'){node.status='done';node.output=typeof event.result==='string'?event.result:JSON.stringify(event.result,null,2);if(state.active===node)state.active=null;setStatus(`${state.nodes.length} node${state.nodes.length===1?'':'s'} · complete`,'done');render()}
  }

  function inspectToolPanel(panel){const title=panel.querySelector('.tool-activity-title')?.textContent||'';if(!/codemode\.run_code/i.test(title)||state.sawEvent)return;const output=panel.querySelector('.tool-activity-output')?.textContent||'';const code=extractCode(output);if(!state.active)addNode('Execute CodeMode program','codemode.run_code',code);else if(code&&!state.active.code){state.active.code=code;render()}if(output&&output!=='Running…'&&output!=='No output returned')finishNode(output)}
  function observe(){const messages=document.getElementById('messages');if(!messages)return;const observer=new MutationObserver(mutations=>{for(const mutation of mutations)for(const node of mutation.addedNodes){if(!(node instanceof Element))continue;node.querySelectorAll?.('.tool-activity').forEach(inspectToolPanel);if(node.matches?.('.tool-activity'))inspectToolPanel(node)}messages.querySelectorAll('.tool-activity').forEach(inspectToolPanel)});observer.observe(messages,{childList:true,subtree:true,characterData:true})}
  function hookComposer(){document.getElementById('composer')?.addEventListener('submit',reset,{capture:true})}
  function init(){ensurePanel();observe();hookComposer()}
  return {init,reset,addNode,finishNode,fail,onToolEvent};
})();
document.readyState==='loading'?document.addEventListener('DOMContentLoaded',()=>CodeModeUI.init()):CodeModeUI.init();
