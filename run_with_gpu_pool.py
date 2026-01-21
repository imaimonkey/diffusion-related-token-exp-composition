"""
완전 자동화된 Work Stealing GPU 스케줄러

모든 것이 자동:
- GPU 개수: Slurm 할당 자동 감지
- 청크 크기: GPU 개수와 샘플 수로 자동 계산
- 작업 분배: 동적 work stealing
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import threading
import queue

class GPUPool:
    """GPU 풀 관리자"""
    def __init__(self, gpu_ids: List[int]):
        self.available = queue.Queue()
        for gpu_id in gpu_ids:
            self.available.put(gpu_id)
        self.total = len(gpu_ids)
    
    def acquire(self) -> int:
        """GPU 1개 획득"""
        return self.available.get()
    
    def release(self, gpu_id: int):
        """GPU 반환"""
        self.available.put(gpu_id)
    
    def size(self) -> int:
        return self.total


class WorkChunk:
    """작업 청크"""
    def __init__(self, method: str, dataset: str, model: str, run_id: str,
                 results_dir: Path, shard_idx: int, num_shards: int, 
                 total_samples: Optional[int], override_config: str = "{}"):
        self.method = method
        self.dataset = dataset
        self.model = model
        self.run_id = run_id
        self.results_dir = results_dir
        self.shard_idx = shard_idx
        self.num_shards = num_shards
        self.total_samples = total_samples
        self.override_config = override_config
        
        self.gpu_id = None
        self.start_time = None
        self.end_time = None
        self.success = False
    
    def __repr__(self):
        dataset_short = self.dataset.split('/')[-1]
        return f"{self.method}@{dataset_short} chunk{self.shard_idx}/{self.num_shards}"
    
    def run(self, gpu_id: int) -> bool:
        """청크 실행"""
        self.gpu_id = gpu_id
        self.start_time = datetime.now()
        
        cmd = [
            "uv", "run", "python", "run_experiments.py",
            "--model", self.model,
            "--dataset", self.dataset,
            "--methods", self.method,
            "--results_dir", str(self.results_dir),
            "--run_id", self.run_id,
            "--shard_index", str(self.shard_idx),
            "--num_shards", str(self.num_shards),
            "--override_config", self.override_config
        ]
        
        if self.total_samples:
            cmd.extend(["--num_samples", str(self.total_samples)])
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output
        
        try:
            # 정상 실행: stdout/stderr를 직접 연결하여 실시간 진행률 표시
            result = subprocess.run(cmd, env=env, check=True)
            self.success = True
            return True
        except subprocess.CalledProcessError as e:
            self.success = False
            # 오류 발생 시: 오류 정보를 캡처하여 출력
            print(f"\n{'='*60}", flush=True)
            print(f"[ERROR] Chunk failed: {self}", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"Command: {' '.join(cmd)}", flush=True)
            print(f"Exit code: {e.returncode}", flush=True)
            
            # 오류 메시지를 다시 실행하여 캡처 (이미 실패했으므로 빠름)
            try:
                err_result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if err_result.stdout:
                    print(f"\nSTDOUT:\n{err_result.stdout}", flush=True)
                if err_result.stderr:
                    print(f"\nSTDERR:\n{err_result.stderr}", flush=True)
            except:
                pass
            
            print(f"{'='*60}\n", flush=True)
            return False
        finally:
            self.end_time = datetime.now()


def worker_thread(task_queue: queue.Queue, gpu_pool: GPUPool, results: List[WorkChunk]):
    """워커 스레드"""
    while True:
        try:
            chunk = task_queue.get(timeout=1)
        except queue.Empty:
            break
        
        gpu_id = gpu_pool.acquire()
        
        try:
            print(f"[GPU {gpu_id}] Starting: {chunk}", flush=True)
            chunk.run(gpu_id)
            status = "✓" if chunk.success else "✗"
            duration = (chunk.end_time - chunk.start_time).total_seconds() / 60
            print(f"{status} [GPU {gpu_id}] Completed: {chunk} ({duration:.1f}m)", flush=True)
            results.append(chunk)
        finally:
            gpu_pool.release(gpu_id)
            task_queue.task_done()


def calculate_optimal_chunks(total_samples: int, num_gpus: int) -> int:
    """
    최적의 청크 수 자동 계산
    
    목표: GPU당 2-3개 청크 (동적 재할당 가능하도록)
    """
    if total_samples is None:
        return num_gpus  # 전체 데이터셋이면 GPU 개수만큼
    
    # GPU당 2-3개 청크
    target_chunks = num_gpus * 2
    
    # 청크당 최소 50개, 최대 500개 샘플
    chunk_size = total_samples // target_chunks
    chunk_size = max(50, min(500, chunk_size))
    
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    return max(num_gpus, num_chunks)  # 최소한 GPU 개수만큼


def merge_results(chunks: List[WorkChunk], results_dir: Path):
    """청크 결과 병합"""
    # 작업별로 그룹화
    task_groups = {}
    for chunk in chunks:
        key = (chunk.method, chunk.dataset)
        if key not in task_groups:
            task_groups[key] = []
        task_groups[key].append(chunk)
    
    # 각 작업별로 병합
    for (method, dataset), chunk_list in task_groups.items():
        dataset_short = dataset.replace("/", "_").replace("openai_", "").replace("ScaleAI_", "")
        base_filename = f"{dataset_short}_{method}"
        
        # JSONL 파일 병합
        merged_results = []
        for chunk in sorted(chunk_list, key=lambda c: c.shard_idx):
            shard_file = results_dir / f"{base_filename}_shard{chunk.shard_idx}.jsonl"
            if shard_file.exists():
                with open(shard_file, "r", encoding="utf-8") as f:
                    for line in f:
                        merged_results.append(json.loads(line))
                shard_file.unlink()
        
        # 병합된 결과 저장
        if merged_results:
            merged_file = results_dir / f"{base_filename}.jsonl"
            with open(merged_file, "w", encoding="utf-8") as f:
                for result in merged_results:
                    f.write(json.dumps(result, ensure_ascii=True) + "\n")
            
            # Summary 생성
            total_correct = sum(1 for r in merged_results if r.get("correct", False))
            total_samples = len(merged_results)
            avg_nfe = sum(r.get("steps", 0) for r in merged_results) / total_samples if total_samples > 0 else 0
            
            summary = {
                "method": method,
                "dataset": dataset,
                "total_samples": total_samples,
                "accuracy": 100.0 * total_correct / total_samples if total_samples > 0 else 0.0,
                "correct_count": total_correct,
                "avg_nfe": float(avg_nfe),
                "median_nfe": float(sorted([r.get("steps", 0) for r in merged_results])[total_samples // 2]) if total_samples > 0 else 0.0,
                "work_stealing": True,
                "num_chunks": len(chunk_list),
            }
            
            summary_file = results_dir / f"summary_{method}.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Merged {len(chunk_list)} chunks for {method}@{dataset_short}: {total_correct}/{total_samples} correct", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fully automated work stealing scheduler")
    parser.add_argument("--gpu_pool", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--methods", type=str, nargs="+", required=True)
    parser.add_argument("--num_samples", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--override_config", type=str, default="{}", help="JSON string for config overrides")
    
    args = parser.parse_args()
    
    # GPU 설정
    gpu_ids = [int(x.strip()) for x in args.gpu_pool.split(',')]
    num_gpus = len(gpu_ids)
    
    # 샘플 수
    num_samples = int(args.num_samples) if args.num_samples and args.num_samples.strip() else None
    
    # 결과 디렉터리
    results_dir = Path(args.results_dir) / args.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}", flush=True)
    print("완전 자동화 Work Stealing 스케줄러", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"GPU Pool: {gpu_ids} ({num_gpus} GPUs)", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"Methods: {args.methods}", flush=True)
    print(f"Samples: {num_samples or 'ALL'}", flush=True)
    print(f"Overrides: {args.override_config}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 작업 생성 및 청킹
    all_chunks = []
    for dataset in args.datasets:
        for method in args.methods:
            # 최적의 청크 수 자동 계산
            num_chunks = calculate_optimal_chunks(num_samples, num_gpus)
            
            print(f"{method}@{dataset.split('/')[-1]}: {num_chunks} chunks (auto-calculated)", flush=True)
            
            # 청크 생성
            for shard_idx in range(num_chunks):
                chunk = WorkChunk(
                    method=method,
                    dataset=dataset,
                    model=args.model,
                    run_id=args.run_id,
                    results_dir=results_dir,
                    shard_idx=shard_idx,
                    num_shards=num_chunks,
                    total_samples=num_samples,
                    override_config=args.override_config
                )
                all_chunks.append(chunk)
    
    print(f"\nTotal chunks: {len(all_chunks)}\n", flush=True)
    
    # GPU 풀 및 작업 큐
    gpu_pool = GPUPool(gpu_ids)
    task_queue = queue.Queue()
    for chunk in all_chunks:
        task_queue.put(chunk)
    
    # 워커 스레드 시작
    results = []
    threads = []
    for _ in range(num_gpus):
        t = threading.Thread(target=worker_thread, args=(task_queue, gpu_pool, results))
        t.start()
        threads.append(t)
    
    # 완료 대기
    task_queue.join()
    for t in threads:
        t.join()
    
    # 결과 병합
    print(f"\n청크 결과 병합 중...", flush=True)
    merge_results(results, results_dir)
    
    # 요약
    print(f"\n{'='*60}", flush=True)
    print("실험 완료", flush=True)
    print(f"{'='*60}", flush=True)
    success_count = sum(1 for c in results if c.success)
    print(f"Total chunks: {len(results)}", flush=True)
    print(f"Success: {success_count}/{len(results)}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
