import subprocess

def get_cuda_version():
    try:
        result = subprocess.run(['which', 'nvcc'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        lines = output.split('\n')

        return lines
    except FileNotFoundError:
        return None

cuda_version = get_cuda_version()
if cuda_version:
    print("CUDA version:", cuda_version)
else:
    print("CUDA is not installed or not found.")
