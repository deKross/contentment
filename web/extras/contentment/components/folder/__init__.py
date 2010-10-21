# encoding: utf-8

from web.extras.contentment import release
from web.extras.contentment.api import IComponent


__all__ = ['FolderComponent', 'controller', 'model', 'templates']
log = __import__('logging').getLogger(__name__)


class FolderComponent(IComponent):
    title = "Folder"
    summary = "A simple container."
    description = None
    icon = 'base-folder'
    group = "Basic Types"
    
    version = release.version
    author = release.author
    email = release.email
    url = release.url
    copyright = release.copyright
    license = release.license
    
    @property
    def model(self):
        from web.extras.contentment.components.folder import model
        return super(FolderComponent, self).model(model)
    
    @property
    def controller(self):
        from web.extras.contentment.components.folder.controller import FolderController
        FolderController._component = self
        return FolderController
