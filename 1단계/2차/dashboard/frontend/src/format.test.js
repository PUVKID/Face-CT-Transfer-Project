import test from 'node:test';
import assert from 'node:assert/strict';
import { metricText, numberText, percent, safeUrl, validateStatus, bytes, statusText, verificationText, processText, coverageText, failedImagesText, throughputLabel } from './format.js';

test('미측정과 실제 영 값을 구분한다', () => {
  assert.equal(numberText(null), '미측정');
  assert.equal(numberText(0), '0');
  assert.equal(metricText({ value: null, unit: '%' }), '미측정');
  assert.equal(metricText({ value: 0, unit: '%' }), '0%');
  assert.equal(metricText({ value: 97.84, unit: '%' }), '97.84%');
});
test('유한한 실제 진행률만 표시한다', () => {
  assert.equal(percent(null), null);
  assert.equal(percent(-1), null);
  assert.equal(percent(101), null);
  assert.equal(percent(0), 0);
  assert.equal(percent(55.25), 55.25);
  assert.equal(numberText(Infinity), '미측정');
});
test('출처 링크에 실행 가능한 URL을 허용하지 않는다', () => {
  assert.equal(safeUrl('javascript:alert(1)'), null);
  assert.equal(safeUrl('file:///private/log'), null);
  assert.equal(safeUrl('https://example.org/model'), 'https://example.org/model');
});
test('다른 스키마와 잘못된 상태를 정상 데이터로 취급하지 않는다', () => {
  assert.throws(() => validateStatus({ schema_version: 2 }));
  assert.throws(() => validateStatus(null));
  assert.throws(() => validateStatus({ schema_version: 1, status: 'invented' }));
  assert.equal(validateStatus({ schema_version: 1, status: 'waiting' }).status, 'waiting');
});
test('사용자 중단과 기다림을 구분한다', () => {
  assert.equal(statusText('user_stopped'), '사용자 중단 · 보존');
  assert.equal(statusText('waiting'), '대기');
  assert.equal(bytes(0), '0 B');
  assert.equal(bytes(null), '미확인');
});
test('최초 획득 로컬 해시를 공식 체크섬 대조로 오표기하지 않는다', () => {
  assert.equal(verificationText({ verified:true, official_digest_verified:false, verification_basis:'official_size_and_first_acquisition_sha256_no_published_digest' }), '최초 로컬 SHA-256 고정 · 공식 해시 미제공');
  assert.equal(verificationText({ verified:false, official_digest_verified:false, verification_basis:'official_size_and_first_acquisition_sha256_no_published_digest' }), '최초 획득 해시 고정 전 · 공식 해시 미제공');
  assert.equal(verificationText({ official_digest_verified:true }), '제공된 공식 해시 대조 완료');
});
test('프로세스 생존 true false null을 구분한다', () => {
  assert.equal(processText(true), '생존 확인');
  assert.equal(processText(false), '종료 확인');
  assert.equal(processText(null), '생존 미확인');
  assert.equal(processText(undefined), '생존 미확인');
});
test('평가조건 미충족을 알려진 평가 상태로 표시한다', () => {
  assert.equal(statusText('protocol_unmet'), '평가조건 미충족');
  assert.equal(statusText('ready'), '준비 완료');
  assert.equal(statusText(undefined), '미확인');
});
test('검출 coverage 분율과 미확인 값을 구분한다', () => {
  assert.equal(coverageText(1), '100%');
  assert.equal(coverageText(0.875), '87.5%');
  assert.equal(coverageText(0), '0%');
  for (const value of [null, undefined, -0.1, 1.1, Infinity, NaN, '1']) assert.equal(coverageText(value), '미확인');
});
test('검출 실패의 실제 정수 개수와 미확인 값을 구분한다', () => {
  assert.equal(failedImagesText(0), '0개');
  assert.equal(failedImagesText(12), '12개');
  for (const value of [null, undefined, -1, 1.5, Infinity, NaN, '0', []]) assert.equal(failedImagesText(value), '미확인');
});
test('바이트 단계 처리량은 초당 바이트 수신으로 표시한다', () => {
  assert.equal(throughputLabel('bytes'), '바이트 수신 속도 (바이트/초)');
});
test('영상 단계로 전환하면 초당 영상 처리 레이블로 바뀐다', () => {
  assert.equal(throughputLabel('images'), '영상 처리 속도 (영상/초)');
  assert.notEqual(throughputLabel('images'), throughputLabel('bytes'));
});
test('없는 처리량 단위나 지원하지 않는 단위는 미확인으로 표시한다', () => {
  for (const unit of [null, undefined, '', 'unknown', 'pairs']) assert.equal(throughputLabel(unit), '처리 속도 (단위 미확인)');
});
