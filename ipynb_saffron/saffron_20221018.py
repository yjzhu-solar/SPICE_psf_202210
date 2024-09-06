from saffron.manager.manager import Manager
from saffron.utils import get_input_template
import sospice
import sunpy.map

session = Manager("./saffron_json/input_1018.json")
session.build_files_list()
session.build_rasters()

session.run_preparations()
session.fit_manager()
