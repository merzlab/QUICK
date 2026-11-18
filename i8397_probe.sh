#!/usr/bin/env bash
# i8397 read-only GPU/host probe
set +x
echo "=== I8397_PROBE_BEGIN host=$(hostname) date=$(date -u +%FT%TZ) ==="
id
hostname
uname -a
echo "--- nvidia ---"
(nvidia-smi -L 2>&1 | head -20) || true
(nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader 2>&1 | head -20) || true
echo "--- rocm ---"
(rocm-smi --showproductname 2>&1 | head -25) || true
ls -l /dev/nvidia* /dev/kfd 2>&1 | head -20
echo "--- cpu-mem ---"
nproc 2>/dev/null; (free -g 2>/dev/null | head -3) || true
echo "--- IMDS aws ---"
TOKEN=$(curl -s -m 2 -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>&1)
echo "aws_token_len=${#TOKEN}"
echo -n "aws_instance_id="; curl -s -m 3 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id 2>&1 | head -c 120; echo
echo -n "aws_instance_type="; curl -s -m 3 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type 2>&1 | head -c 120; echo
echo -n "aws_iam_role="; curl -s -m 3 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/ 2>&1 | head -c 200; echo
echo "--- IMDS gcp ---"
echo -n "gcp_instance="; curl -s -m 3 -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/name" 2>&1 | head -c 120; echo
echo -n "gcp_sa="; curl -s -m 3 -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/" 2>&1 | head -c 200; echo
echo "--- IMDS azure ---"
echo -n "azure_name="; curl -s -m 3 -H "Metadata: true" "http://169.254.169.254/metadata/instance/compute/name?api-version=2021-02-01&format=text" 2>&1 | head -c 120; echo
echo "--- IMDS alibaba ---"
echo -n "ali_instance="; curl -s -m 3 "http://100.100.100.200/latest/meta-data/instance-id" 2>&1 | head -c 120; echo
echo "=== I8397_PROBE_END host=$(hostname) ==="
