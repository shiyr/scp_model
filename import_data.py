import sys
import logging
import pandas as pd
from xlrd import XLRDError
import os.path
from StringIO import StringIO
import utils
import db_utils
import data_model as dm
from sqlalchemy import inspect
from sqlalchemy_utils import get_table_classes, add_all

logger = logging.getLogger("import_data")


class FrameImporter(object):
    def to_db(self):
        table_classes = get_table_classes(dm)
        print 'table_classes', table_classes
        print 'start getting session'
        session = db_utils.get_session(dm=dm)
        print 'complete getting session'
        optimization_run = dm.OptimizationRun()
        session.add(optimization_run)
        session.flush()
        run_id = optimization_run.id
        logger.info("reading data into run_id %d", run_id)

        with self.input_source as input_source:
            for table in dm.tables_in_order:
                tablename = table.name
                if tablename.find('_result') >= 0:
                    logger.info("skippping result table %s", tablename)
                    continue
                df = self.get_frame(input_source, tablename)
                if df is None:
                    logger.warn("no sheet corresponding to %s, table will be empty", tablename)
                else:
                    df['run_id'] = run_id
                    cls = table_classes[tablename]
                    df_columns = df.columns
                    tb_columns = inspect(cls).columns.keys()
                    columns = [c for c in df_columns if c in tb_columns]
                    print df_columns
                    print tb_columns
                    print columns
                    print df[columns]
                    
                    objects = [cls(**{name: row[name] for name in columns if not pd.isnull(row[name])})
                               for idx, row in df.iterrows()]
                    logger.info("adding %d %s objects to database", len(df), tablename)
                    try:
                        add_all(session, objects)
                    except Exception as ex:
                        print ex
                        for obj in objects:
                            print obj
                        raise


class ExcelWithTableSheetsImporter(FrameImporter):
    def __init__(self, filename):
        self.filename = filename
        logger.info("reading data from file %s", filename)

    @utils.lazy_property
    def input_source(self):
        return pd.ExcelFile(self.filename)

    def get_frame(self, excel_file, tablename):
        try:
            return excel_file.parse(tablename)
        except XLRDError:
            return None


class ZipOfCSVImporter(FrameImporter):
    def __init__(self, filename):
        self.filename = filename
        logger.info("reading data from file %s", filename)
        self.map_csv_to_full_path = {os.path.basename(name): name for name in self.input_source.namelist()}
    
    @utils.lazy_property
    def input_source(self):
        return zipfile.ZipFile(self.filename, 'r')

    def get_frame(self, zf, tablename):
        try:
            filename = self.map_csv_to_full_path[tablename + ".csv"]
            df = pd.read_csv(StringIO(zf.read(filename)), na_values=[''], keep_default_na=False)
            return df
        except KeyError:
            return None


@utils.logged
def import_data(filename):
    if utils.is_excel_file(filename):
        ExcelWithTableSheetsImporter(filename).to_db()
    else:
        ZipOfCSVImporter(filename).to_db()


def main():
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
    else:
        raise RuntimeError("usage: import_data.py <input file name>")
    import_data(input_filename)


if __name__ == "__main__":
    main()