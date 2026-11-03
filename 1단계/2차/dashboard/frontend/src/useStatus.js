import { useCallback, useEffect, useRef, useState } from 'react';
import { validateStatus } from './format';

export function useStatus() {
  const [data,setData] = useState(null), [error,setError] = useState(''), [loading,setLoading] = useState(true);
  const pending = useRef(null), mounted = useRef(true);
  const refresh = useCallback(async () => {
    if (pending.current) return;
    const controller = new AbortController(); pending.current = controller;
    const timer = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch('/api/status', { signal:controller.signal, cache:'no-store' });
      if (!response.ok) throw new Error(`상태 연결 오류 (${response.status})`);
      const next = validateStatus(await response.json());
      if (mounted.current) { setData(next); setError(''); }
    } catch (failure) {
      if (mounted.current) setError(failure.name === 'AbortError' ? '상태 서버 응답 시간이 초과되었습니다.' : failure.message);
    } finally { clearTimeout(timer); pending.current = null; if (mounted.current) setLoading(false); }
  }, []);
  useEffect(() => {
    mounted.current = true; refresh(); const timer = setInterval(refresh, 3000);
    return () => { mounted.current = false; clearInterval(timer); pending.current?.abort(); };
  }, [refresh]);
  return { data,error,loading,refresh };
}

export function useLogs(source) {
  const [data,setData] = useState(null), [error,setError] = useState(''), [loading,setLoading] = useState(false);
  const pending = useRef(null), mounted = useRef(true);
  const refresh = useCallback(async () => {
    if (pending.current) return;
    const controller = new AbortController(); pending.current = controller;
    let timedOut = false;
    setLoading(true); const timer = setTimeout(() => { timedOut = true; controller.abort(); }, 8000);
    try {
      const response = await fetch(`/api/logs?id=${encodeURIComponent(source)}&limit=100`, { cache:'no-store',signal:controller.signal });
      if (!response.ok) throw new Error(`로그 연결 오류 (${response.status})`);
      const next = await response.json();
      if (!Array.isArray(next.lines) || next.lines.some(line => typeof line !== 'string')) throw new Error('로그 응답 형식을 확인할 수 없습니다.');
      if (mounted.current && pending.current === controller) { setData(next); setError(''); }
    } catch (failure) { if (mounted.current && pending.current === controller) setError(timedOut ? '로그 서버 응답 시간이 초과되었습니다.' : failure.message); }
    finally { clearTimeout(timer); if (pending.current === controller) { pending.current = null; if (mounted.current) setLoading(false); } }
  }, [source]);
  useEffect(() => { mounted.current=true; setData(null); setError(''); refresh(); const timer=setInterval(refresh,5000); return () => { mounted.current=false; clearInterval(timer); const controller=pending.current; pending.current=null; controller?.abort(); }; },[refresh]);
  return { data,error,loading,refresh };
}
