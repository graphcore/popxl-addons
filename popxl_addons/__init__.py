# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from .module import *
from .graph_cache import *
from .route_tensor import *
from .runner import *
from .ops import *
from .transforms import *
from .streams import *
from .utils import *

import git
try:
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    GIT_COMMIT = repo.git.rev_parse(sha, short=7)
    del repo, sha
except:
    GIT_COMMIT = None
del git
