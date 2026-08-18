const WORKFLOW_HISTORY_VERSION='v2';

function workflowSnapshot(thinking){
  if(!thinking?.workflow)return [];
  return Array.from(thinking.workflow.stepsMap.values()).map(panel=>({name:panel.name,output:panel.output.textContent||'No output returned'}));
}

function ensureWorkflowHistoryEntry(session,thinking){
  const all=loadHistoryJSON();
  const items=Array.isArray(all[session])?all[session]:[];
  let entry=null;
  for(let i=items.length-1;i>=0;i--){
    if(items[i]?.role==='assistant'&&items[i]?.workflow){entry=items[i];break}
  }
  if(!entry){
    entry={role:'assistant',text:'',timestamp:new Date().toISOString(),workflow:{version:WORKFLOW_HISTORY_VERSION,steps:[],status:'running'}};
    items.push(entry);
  }
  all[session]=items;
  saveHistoryJSON(all);
  return entry;
}

function persistWorkflow(session,thinking,status='running'){
  const steps=workflowSnapshot(thinking);
  if(!steps.length)return;
  const all=loadHistoryJSON();
  const items=Array.isArray(all[session])?all[session]:[];
  let entry=null;
  for(let i=items.length-1;i>=0;i--){
    if(items[i]?.role==='assistant'&&items[i]?.workflow){entry=items[i];break}
  }
  if(!entry){
    entry={role:'assistant',text:'',timestamp:new Date().toISOString(),workflow:{version:WORKFLOW_HISTORY_VERSION,steps:[],status}};
    items.push(entry);
  }
  entry.workflow={version:WORKFLOW_HISTORY_VERSION,steps,status};
  entry.timestamp=entry.timestamp||new Date().toISOString();
  all[session]=items;
  saveHistoryJSON(all);
  renderHistory();
}

function renderPersistedWorkflow(bubble,workflow){
  if(!workflow?.steps?.length)return;
  const thinking={
    row:{classList:{add(){}}},
    bubble,
    label:{textContent:''},
    dots:{style:{display:''}},
    tool:null,
    workflow:null
  };
  ensureWorkflowPanel(thinking);
  for(const step of workflow.steps){
    const panel=ensureToolPanel(thinking,step.name);
    if(panel){
      panel.output.textContent=String(step.output??'No output returned');
      panel.output.dataset.ready='true';
    }
  }
  finishToolPanel(thinking);
  bubble.replaceChildren(thinking.workflow.container);
  bubble.classList.add('thinking-bubble');
}

function renderSessionWithWorkflow(id){
  messages.replaceChildren();
  const items=sessionHistory(id);
  if(!items.length){
    messages.innerHTML='<div class="welcome"><div class="welcome-mark">λ</div><h1>Build with your agent.</h1><p>Start a conversation. Messages are persisted as JSON in this browser.</p><div class="suggestions"><button data-prompt="What skills are available?">List skills</button><button data-prompt="What tools are available?">List tools</button></div></div>';
    document.querySelectorAll('[data-prompt]').forEach(b=>b.addEventListener('click',()=>{input.value=b.dataset.prompt;input.focus();input.dispatchEvent(new Event('input'))}));
    return;
  }
  for(const item of items){
    const bubble=addMessage(item.role,item.text,false);
    if(item.role==='assistant'&&item.workflow)renderPersistedWorkflow(bubble,item.workflow);
  }
  messages.scrollTop=messages.scrollHeight;
}

renderSession=renderSessionWithWorkflow;

const originalStreamChat=streamChat;
streamChat=async function(session,message,bubble,thinking){
  ensureWorkflowHistoryEntry(session,thinking);
  let stopped=false;
  const timer=setInterval(()=>{if(!stopped)persistWorkflow(session,thinking,'running')},200);
  try{return await originalStreamChat(session,message,bubble,thinking)}
  finally{
    stopped=true;
    clearInterval(timer);
    persistWorkflow(session,thinking,'complete');
  }
};