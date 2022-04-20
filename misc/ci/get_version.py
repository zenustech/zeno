import datetime
d = datetime.datetime.now()
print('::set-output name=version::{}.{}.{}'.format(d.year, d.month, d.day))
