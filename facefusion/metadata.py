from typing import Optional

METADATA =\
{
	'name': 'morphflow-swap-engine',
	'description': 'MorphFlow face swap engine baseline for modular engine migration',
	'version': '0.1.0',
	'license': 'OpenRAIL-AS',
	'author': 'MorphFlow',
	'url': ''
}


def get(key : str) -> Optional[str]:
	return METADATA.get(key)
