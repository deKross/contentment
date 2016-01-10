# encoding: utf-8

from marrow.package.loader import load
from marrow.package.host import PluginManager

from web.component.asset.model import Asset


log = __import__('logging').getLogger(__name__)


def indent(context, lines, padding='\t'):
	return padding + ('\n' + padding).join(lines.split('\n'))


MAP = {
		'localhost': ('career.nse-automatech.com', 'en'),
		
		# NSE Automatech
		# Testing URLs
		'en.nse.illico.cleverapps.io': ('career.nse-automatech.com', 'en'),
		'fr.nse.illico.cleverapps.io': ('career.nse-automatech.com', 'fr'),
		# Production URLs
		'career.nse-automatech.com': ('career.nse-automatech.com', 'en'),
		'carrieres.nse-automatech.com': ('career.nse-automatech.com', 'fr'),
		
	}


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
		dom = context.request.host.partition(':')[0]
		parts = MAP.get(dom, (dom, 'en'))
		context.domain = parts[0]
		context.lang = parts[1]
		context.croot = Asset.objects.nearest('/' + context.domain)
		
		if context.croot:
			context.theme = load(context.croot.properties.theme + ':page')
		else:
			context.theme = load('web.theme.bootstrap.base:page')
		
		log.info("Prepared context.", extra=dict(domain=[dom, context.domain], lang=context.lang, root=repr(context.croot), theme=repr(context.theme)))
