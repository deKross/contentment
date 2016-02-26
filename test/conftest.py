# encoding: utf-8
import pytest

from web.contentment.taxonomy import Taxonomy


@pytest.fixture(scope='module')
def connection(request):
	from mongoengine import connect
	from .fixture import apply

	connection = connect('testing')
	apply()

	request.addfinalizer(lambda: connection.drop_database('testing'))

	return connection


@pytest.fixture
def root(connection):
	from web.component.asset.model import Asset
	return Asset.objects.get(path='/careers.illicohodes.com')


@pytest.fixture(autouse=True)
def contentment_cache(connection):
	from web.ext.contentment import ContentmentCacheExtension
	ContentmentCacheExtension().before({})


@pytest.fixture
def models(connection):
	from collections import namedtuple

	Models = namedtuple('Models', 'parent child1 child2')
	TModel.objects.delete()
	return Models(
		TModel.objects.create(name='parent', path='/'),
		TModel.objects.create(name='child1', path='1'),
		TModel.objects.create(name='child2', path='2'),
	)


class TModel(Taxonomy):
	pass
