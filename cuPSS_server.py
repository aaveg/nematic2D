import subprocess
import numpy as np

## compile command '     nvcc nematic.cu -lcufft -lfftw3f -lcurand -lcupss -O2 -I/home/aaveg/cuPSS/inc -o nematic   '

class cuPSS_server(object):
    def __init__(self, server_name) -> None:
        self.server_name = server_name

        # _process is the instance of the current process
        self._process = None
        self._pname = None
        assert self.compile(), "error while compiling. server name used: {}".format(server_name)


    def compile(self):
        # compile cuPSS .cpp file
        self._pname =  self.server_name.split('.')[0]
        # print(self._pname)
        print("Compiling cuPSS server with user defined parameters...")

        try:
            subprocess.run(['nvcc', self.server_name,  '-lcufft', '-lfftw3f', '-lcurand', '-lcupss', '-O2', '-I/home/aaveg/cuPSS/inc', '-o', self._pname],
                            capture_output = True,
                            check = True,
                            text = True
                            )
        except subprocess.CalledProcessError as e:
            print('Error while compiling cuPSS solver. ')
            print(e.stderr)
            return None

        return True

    def run(self):
        self._process = self._run_instance()
        assert self._clean_cupss_intro(), 'Did not receive the "begin" command'
        return True

    def write(self, data_in):
        # val = '{}\n'.format(data_in)
        # self.process.stdin.write(val.encode())
        np.savetxt(self._process.stdin,data_in)
        # print(data_in, data_in.dtype)
        self._process.stdin.flush()

    def read(self):
        data =  self._process.stdout.readline().strip().decode()
        # print(data)
        return data 

    def reset(self):
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
        assert self.run()

    def close(self):
        print('closing connection to cuPSS server')
        if self._process is not None:
            self._process.terminate()

    def _run_instance(self):
        assert self._pname is not None, "process name (_pname) was set. Should be defined during self.compile()"
        process = subprocess.Popen([self._pname], 
                                  stdout = subprocess.PIPE, 
                                  stdin  = subprocess.PIPE,
                                  stderr = None
                                  )
        # print(process.returncode)
        return process
    
    def _clean_cupss_intro(self):
        for _ in range(200):
            intro = self.read()
            # print(intro)
            if intro == 'begin': 
                return True
            
        self.close()
        return False
