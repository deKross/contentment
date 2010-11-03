# encoding: utf-8

"""Asset model.

All asset types must descend from this class.
"""

import re
import web
import textile

import mongoengine as db

from web.extras.contentment.components.asset.model import Asset

from widgets import fields


log = __import__('logging').getLogger(__name__)
__all__ = ['Page']



class Page(Asset):
    _indexable = ['_content']
    _widgets = fields
    
    default = db.StringField(default="view:page", max_length=128)
    
    content = db.StringField()
    engine = db.StringField(max_length=250, default="textile")
    template = db.StringField(max_length=250)
    related = db.ListField(db.ReferenceField(Asset), default=list)
    
    @property
    def rendered(self):
        @web.core.cache.cache('page.content', expires=3600)
        def cache(name, date):
            # TODO: Make this plugin-able.
            
            if self.engine == 'textile':
                return textile.Textile().textile(self.content, html_type='html')
            
            else:
                return self.content
        
        return cache(self.path, self.modified if self.modified else self.created)
    
    @property
    def _content(self):
        """A HTML-stripped version of the rendered content for indexing."""
        
        from HTMLParser import HTMLParser
        
        class Stripper(HTMLParser):
            def __init__(self):
                self.reset()
                self.fed = []
            
            def handle_data(self, d):
                self.fed.append(d)
            
            def get_data(self):
                return ''.join(self.fed)
        
        s = Stripper()
        s.feed(self.rendered)
        
        return s.get_data()
