export const finite = value => typeof value === 'number' && Number.isFinite(value);
export const list = value => Array.isArray(value) ? value : [];
export const processText = alive => alive === true ? '생존 확인' : alive === false ? '종료 확인' : '생존 미확인';
export const coverageText = value => finite(value) && value >= 0 && value <= 1 ? `${numberText(value * 100)}%` : '미확인';
export const failedImagesText = value => Number.isSafeInteger(value) && value >= 0 ? `${numberText(value)}개` : '미확인';
export const throughputLabel = unit => unit === 'bytes' ? '바이트 수신 속도 (바이트/초)' : unit === 'images' ? '영상 처리 속도 (영상/초)' : '처리 속도 (단위 미확인)';
export const numberText = value => finite(value) ? new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 2 }).format(value) : '미측정';
export const percent = value => finite(value) && value >= 0 && value <= 100 ? value : null;
export const metricText = metric => finite(metric?.value) ? `${numberText(metric.value)}${metric.unit || ''}` : '미측정';
export function safeUrl(value) {
  try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) ? url.href : null; } catch { return null; }
}
export function bytes(value) {
  if (!finite(value) || value < 0) return '미확인';
  if (!value) return '0 B';
  const scale = Math.min(4, Math.floor(Math.log(value) / Math.log(1024)));
  return `${numberText(value / 1024 ** scale)} ${['B', 'KiB', 'MiB', 'GiB', 'TiB'][scale]}`;
}
export function statusText(status) {
  return ({ waiting:'대기', pending:'대기', downloading:'수신 중', verifying:'검증 중', ready:'준비 완료', running:'실행 중', active:'진행 중', complete:'완료', paused:'일시 중단', failed:'오류', protocol_unmet:'평가조건 미충족', user_stopped:'사용자 중단 · 보존', planned:'계획', unused:'미사용' })[status] || '미확인';
}
export function validateStatus(value) {
  if (!value || value.schema_version !== 1 || !['waiting','downloading','verifying','ready','running','complete','paused','failed'].includes(value.status)) throw new Error('지원하지 않는 상태 응답입니다.');
  return value;
}
export function verificationText(file) {
  if (file?.official_digest_verified === true) return '제공된 공식 해시 대조 완료';
  if (file?.verification_basis === 'official_size_and_first_acquisition_sha256_no_published_digest') return `${file.verified ? '최초 로컬 SHA-256 고정' : '최초 획득 해시 고정 전'} · 공식 해시 미제공`;
  return file?.verified ? '획득 검증 완료 · 공식 해시 대조 여부 별도 확인' : '획득 검증 대기';
}
export function timeText(value) {
  const time = value ? new Date(value) : null;
  return time && !Number.isNaN(time.valueOf()) ? new Intl.DateTimeFormat('ko-KR', { timeZone:'Asia/Seoul', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false }).format(time) : '관측 전';
}
export function seconds(value, estimate = false) {
  if (!finite(value) || value < 0) return '추정 불가';
  const hours = Math.floor(value / 3600), minutes = Math.floor(value % 3600 / 60);
  return `${estimate ? '약 ' : ''}${hours ? `${hours}시간 ` : ''}${minutes}분 ${Math.floor(value % 60)}초${estimate ? ' · 추정' : ''}`;
}
