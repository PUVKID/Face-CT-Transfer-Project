import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// 실제 hook 본문을 실행하고 React 생명주기·시간·네트워크 경계만 제어합니다.
// 실제 서버나 운영 로그를 수정하지 않는 결정적 지연 응답 회귀입니다.
const source = fs.readFileSync(new URL('./useStatus.js', import.meta.url), 'utf8');
const body = source.slice(source.indexOf('export function useLogs')).replace('export function', 'function');
function harness(plan = () => ({ delay: 10 })) {
  let now=0, sequence=0, index=0;
  const tasks=new Map(), states=[], refs=[], callbacks=[], effects=[], queue=[], requests=[], writes=[];
  const schedule=(fn,delay,repeat=0)=>{const id=++sequence;tasks.set(id,{fn,time:now+delay,repeat});return id;};
  const cancel=id=>tasks.delete(id);
  const context={AbortController,encodeURIComponent,
    useState(initial){const i=index++;if(!(i in states))states[i]=initial;return [states[i],value=>{states[i]=value;writes.push({i,value});}];},
    useRef(initial){const i=index++;return refs[i]??=( {current:initial} );},
    useCallback(fn,deps){const i=index++;const old=callbacks[i];if(!old||deps.some((d,j)=>!Object.is(d,old.deps[j])))callbacks[i]={fn,deps};return callbacks[i].fn;},
    useEffect(fn,deps){const i=index++;const old=effects[i];if(!old||deps.some((d,j)=>!Object.is(d,old.deps[j])))queue.push(()=>{old?.cleanup?.();effects[i]={deps,cleanup:fn()};});},
    setTimeout:(fn,delay)=>schedule(fn,delay),clearTimeout:cancel,
    setInterval:(fn,delay)=>schedule(fn,delay,delay),clearInterval:cancel,
    fetch(url,{signal}){
      const settings=plan(url,requests.length),request={url,signal,aborted:false};requests.push(request);
      return new Promise((resolve,reject)=>{
        const id=schedule(()=>resolve({ok:settings.ok!==false,status:settings.status||200,json:async()=>({source:url.includes('evaluation')?'evaluation':'acquisition',lines:[url]})}),settings.delay??10);
        signal.addEventListener('abort',()=>{request.aborted=true;if(!settings.ignoreAbort){cancel(id);reject(Object.assign(new Error('제어된 취소'),{name:'AbortError'}));}});
      });
    },
  };
  vm.createContext(context);vm.runInContext(body,context);
  const render=value=>{index=0;context.useLogs(value);while(queue.length)queue.shift()();};
  const flush=async()=>{for(let i=0;i<8;i++)await Promise.resolve();};
  const advance=async ms=>{const until=now+ms;while(true){const due=[...tasks].filter(([,t])=>t.time<=until).sort((a,b)=>a[1].time-b[1].time||a[0]-b[0])[0];if(!due)break;const[id,task]=due;now=task.time;if(task.repeat)task.time+=task.repeat;else tasks.delete(id);task.fn();await flush();}now=until;await flush();};
  render('acquisition');
  return {states,requests,writes,render,advance,unmount(){effects.forEach(e=>e?.cleanup?.());}};
}

test('정상 응답을 표시하고 다음 자동 조회를 계속한다',async()=>{
  const h=harness();await h.advance(20);assert.equal(h.states[0].source,'acquisition');assert.equal(h.states[2],false);
  await h.advance(5000);assert.equal(h.requests.length,2);assert.equal(h.requests[0].aborted,false);
});
test('6.5초 응답을 5초 자동 조회가 취소하지 않는다',async()=>{
  const h=harness(()=>({delay:6500}));await h.advance(5000);assert.equal(h.requests.length,1);assert.equal(h.requests[0].aborted,false);
  await h.advance(1500);assert.equal(h.states[0].source,'acquisition');assert.equal(h.states[1],'');assert.equal(h.states[2],false);
});
test('8초 시간 초과를 오류 문구로 표시하고 다음 조회에서 회복한다',async()=>{
  const h=harness((url,index)=>({delay:index===0?9000:10}));await h.advance(8000);
  assert.match(h.states[1],/초과/);assert.equal(h.states[2],false);assert.equal(h.requests[0].aborted,true);
  await h.advance(2010);assert.equal(h.states[0].source,'acquisition');assert.equal(h.states[1],'');
});
test('출처 전환은 이전 요청을 취소하고 새 출처를 즉시 조회한다',async()=>{
  const h=harness(url=>({delay:url.includes('evaluation')?10:6500}));await h.advance(100);h.render('evaluation');
  assert.equal(h.requests.length,2);assert.equal(h.requests[0].aborted,true);await h.advance(20);assert.equal(h.states[0].source,'evaluation');
});
test('취소를 무시하고 늦게 도착한 이전 출처 응답을 적용하지 않는다',async()=>{
  const h=harness(url=>({delay:url.includes('evaluation')?10:6500,ignoreAbort:true}));await h.advance(100);h.render('evaluation');
  await h.advance(6500);assert.equal(h.states[0].source,'evaluation');assert.equal(h.states[1],'');
});
test('화면 종료 때 취소하고 늦은 응답으로 상태를 쓰지 않는다',async()=>{
  const h=harness(()=>({delay:2000,ignoreAbort:true}));await h.advance(100);h.unmount();const before=h.writes.length;
  assert.equal(h.requests[0].aborted,true);await h.advance(9000);assert.equal(h.writes.length,before);
});
