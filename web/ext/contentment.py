# encoding: utf-8

from mongoengine import Document

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


class Singleton(type):
	_instances = {}
	def __call__(cls, *args, **kwargs):
		if cls not in cls._instances:
			cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
		return cls._instances[cls]


class ContentmentCache(dict, metaclass=Singleton):
	ATTR_NAME = '_contentment_cache'

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.children_calculated = set()
		self.contents_calculated = set()

		from web.core import local

		setattr(local, self.ATTR_NAME, self)

	@classmethod
	def get_cache(cls):
		import web.core
		try:
			return web.core.local.asset_cache
		except AttributeError:
			return None

	def _invalidate_object(self, doc):
		key = doc.pk
		if key in self.children_calculated:
			self.children_calculated.remove(key)
		if key in self.contents_calculated:
			self.contents_calculated.remove(key)

	def invalidate(self, doc):
		self[str(doc.pk)] = doc
		self._invalidate_object(doc)
		for parent in doc.parents:
			self._invalidate_object(parent)

	def store(self, doc, children=False, content=False):
		self[str(doc.pk)] = doc
		if doc.parent:
			if children:
				self.children_calculated.add(doc.parent.pk)
			if content:
				self.contents_calculated.add(doc.parent.pk)

	def remove(self, doc):
		self.pop(str(doc.pk), None)
		self._invalidate_object(doc)
		for parent in doc.parents:
			self._invalidate_object(parent)

	def clear(self):
		super().clear()
		self.children_calculated.clear()
		self.contents_calculated.clear()


class AssetCacheExtension:
	needs = ['threadlocal']

	def before(self, context):
		ContentmentCache().clear()
