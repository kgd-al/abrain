from pyne_cpp.config import Config as CPPConfig


class Config(CPPConfig):
    dict = {
        k: v for k, v in CPPConfig.__dict__.items() if not k.startswith('__')
    }
