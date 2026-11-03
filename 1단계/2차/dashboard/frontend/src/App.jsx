import React, { lazy, Suspense, useState } from 'react';
import Icon from './Icon';
import { useLogs, useStatus } from './useStatus';
import { list, metricText, numberText, percent, seconds, timeText } from './format';
import { Badge, Downloads, LogsPanel, Measurements, ModelDetails, Resources, Techniques } from './Panels';

const nav=[['overview','home','실험 개요'],['models','layers','모델·벤치마크'],['downloads','database','데이터 수집'],['techniques','settings','기법·설정'],['logs','log','실행 로그']];
const Workflow=lazy(()=>import('./Workflow'));

export default function App() {
  const state=useStatus(),data=state.data;
  const [selected,setSelected]=useState(0),[active,setActive]=useState('overview'),[source,setSource]=useState('acquisition');
  const logs=useLogs(source),steps=list(data?.steps),current=steps[selected];
  const metric=list(data?.model?.reported_metrics).find(item=>item.benchmark?.includes('IJB')&&item.metric?.includes('1e-4'));
  const measured=list(data?.model?.measured_metrics).find(item=>item.benchmark===metric?.benchmark&&item.metric===metric?.metric);
  const next=steps.find(step=>step.status==='waiting');
  const go=id=>{setActive(id);document.getElementById(id)?.scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});};
  const progress=percent(data?.progress?.percent);
  return <div className="app-shell">
    <a className="skip-link" href="#overview">본문으로 이동</a>
    <aside className="sidebar"><div className="brand"><div className="brand-cube"><Icon name="cube" size={32}/></div><div><strong>얼굴 모델 연구</strong><small>{data?.project?.revision||'벤치마크 재현'}</small></div></div><nav aria-label="주 탐색">{nav.map(([id,icon,label])=><button key={id} onClick={()=>go(id)} aria-current={active===id?'page':undefined}><Icon name={icon}/><span>{label}</span></button>)}</nav><div className="sidebar-footer"><span className={`status-dot ${state.error?'failed':data?.status||'waiting'}`}/><span>이 컴퓨터의 읽기 전용 상태</span><small>{data?.project?.revision||'상태 연결 대기'}</small></div></aside>
    <main><header className="page-header"><h1>실험 개요</h1><div className="header-tools"><span>최근 관측 {timeText(data?.updated_at)}<small>한국 시각 · 3초 간격 조회</small></span><button onClick={state.refresh}><Icon name="refresh"/>새로고침</button></div></header>
      {state.error&&<div className="notice error" role="alert">{state.error} {data?'마지막 응답을 표시합니다. 현재 상태로 단정하지 마세요.':'실제 API에 연결되면 상태를 표시합니다.'}</div>}
      {data?.stale&&<div className="notice warning" role="status">상태 갱신이 지연되었습니다. 아래 값은 마지막 관측이며 현재 실행 상태는 확인이 필요합니다.</div>}
      <section id="overview" className="overview-grid"><div className="card task-card"><div className="task-heading"><h2>{data?.current_task?.title||(state.loading?'실제 상태를 불러오고 있습니다':'상태 연결을 기다리고 있습니다')}</h2><p>{data?.current_task?.detail||'상태 서버에서 확인한 작업만 표시합니다.'}</p></div><Suspense fallback={<div className="empty">3D 작업 흐름을 불러오고 있습니다.</div>}><Workflow steps={steps} selected={selected} onSelect={setSelected}/></Suspense><div className="selected-step" aria-live="polite"><div><strong>{current?.title||'단계 정보 대기'}</strong><p>{current?.detail||'API에서 단계 정보를 받으면 선택해 세부 내용을 확인할 수 있습니다.'}</p></div>{current&&<Badge status={current.status}/>}</div></div>
        <aside className="card current-panel"><h2>현재 작업</h2><div className="model-name"><strong>{data?.model?.name||'모델 정보 대기'}</strong><span>{data?.model?.training_data||'사전학습 자료 확인 전'}</span></div><div className="primary-metric"><strong>{metricText(metric)}</strong><p>{metric?`${metric.benchmark} ${metric.metric}`:'IJB-C TAR @ FAR=1e-4'}</p><small>저자 보고 · 직접 재현 전</small></div><div className="direct-metric"><span>직접 측정</span><strong>{metricText(measured)}</strong></div>{data&&<div className="current-status"><Badge status={data.status}/>{progress!==null&&<><strong>{numberText(progress)}%</strong><progress max="100" value={progress} aria-label={`API 작업 진행률 ${numberText(progress)}%`}/></>}<small>{data.progress?.unit?`${numberText(data.progress.completed)} / ${numberText(data.progress.total)} ${data.progress.unit}`:'전체 진행률 미확인'}</small><small>남은 시간 {seconds(data.progress?.eta_seconds,data.progress?.eta_is_estimate)}</small></div>}<button className="next-stage" onClick={()=>{const index=steps.indexOf(next);if(index>=0)setSelected(index);}} disabled={!next}><span>{next?.title||'다음 단계 확인 전'}</span><Icon name="arrow" size={18}/></button></aside>
      </section>
      <div className="lower-grid"><Measurements model={data?.model}/><LogsPanel compact logs={logs} source={source} setSource={setSource} sources={data?.log_sources}/></div>
      <Resources data={data}/>
      {list(data?.notices).length>0&&<section className="notices" aria-label="실제 관측의 한계">{data.notices.map((notice,index)=><p key={index}>{notice}</p>)}</section>}
      <section id="models" className="detail-section"><h2 className="section-title">모델·벤치마크</h2><ModelDetails data={data}/></section>
      <section id="downloads" className="detail-section"><h2 className="section-title">데이터 수집</h2><Downloads data={data}/></section>
      <section id="techniques" className="detail-section"><h2 className="section-title">기법·설정</h2><Techniques data={data}/>{data?.legacy&&<section className="card legacy"><div className="card-heading"><h2>이전 실험 보존</h2><Badge status={data.legacy.status}/></div><strong>{data.legacy.model}</strong><p>{data.legacy.detail}</p></section>}</section>
      <section id="logs" className="detail-section"><h2 className="section-title">실행 로그</h2><LogsPanel logs={logs} source={source} setSource={setSource} sources={data?.log_sources}/></section>
      <footer className="page-footer">이 화면은 실제 상태를 읽습니다. 3D 움직임은 작업 흐름 안내이며 완료율이나 성능을 나타내지 않습니다.</footer>
    </main>
  </div>;
}
