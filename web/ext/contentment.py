# encoding: utf-8

from marrow.package.host import PluginManager


def indent(context, lines, padding='\t'):
	return padding + ('\n' + padding).join(lines.split('\n'))


class ContentmentExtension:
	needs = ('template', )
	
	def __call__(self, context, app):
		def protected_inner(environ, start_response=None):
			try:
				return app(environ, start_response)
			except:
				if __debug__:
					try:
						import pudb; pudb.post_mortem()
					except:
						pass
				raise
		
		return protected_inner
	
	def start(self, context):
		log = __import__('logging').getLogger(__name__)
		log.info("Starting Contentment extension.")
		context.namespace.indent = indent
		
		for asset_type in PluginManager('web.component'):
			log.info("Found asset type: " + repr(asset_type))
		
		# registry.register(render_asset, Asset)
	
	def prepare(self, context):
		context.domain, _, _ = context.request.host.partition(':')


class ContentmentCache(dict):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.children_calculated = set()
		self.contents_calculated = set()

	@classmethod
	def get_cache(cls):
		import web.core
		try:
			return web.core.local.asset_cache
		except AttributeError:
			return None

	# def __getitem__(self, key):
	# 	result = super().__getitem__(key)
	# 	print('From cache: %s' % result)
	# 	return result


class AssetCacheExtension:
	needs = ['threadlocal']

	def before(self, context):
		from web.core import local
		local.asset_cache = ContentmentCache()
