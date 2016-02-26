# encoding: utf-8

import pytest

from web.ext.contentment import ContentmentCache
from web.component.asset.model import Asset


class CursorPatcher:
	__patched = False
	db_count = 0

	@classmethod
	def reset(cls):
		cls.db_count = 0

	@classmethod
	def patch(cls):
		if cls.__patched:
			return
		cls.__patched = True

		from pymongo.cursor import Cursor

		old_getitem = Cursor.__getitem__

		def getitem(self, index):
			result = old_getitem(self, index)
			if not isinstance(result, Cursor):
				cls.db_count += 1
			return result

		Cursor.__getitem__ = getitem

		old_next = Cursor.next

		def next(self):
			result = old_next(self)
			cls.db_count += 1
			return result

		Cursor.next = next
		Cursor.__next__ = next


@pytest.fixture(autouse=True)
def cursor_patcher():
	CursorPatcher.patch()
	CursorPatcher.reset()


@pytest.mark.usefixtures('connection')
class TestContentmentCache:
	def test_cache_counts(self):
		assert len(ContentmentCache()) == 0
		assert len(ContentmentCache().queries) == 0

		root = Asset.objects.get(path='/careers.illicohodes.com')

		assert len(ContentmentCache()) == 1
		assert CursorPatcher.db_count == 1
		list(root.children)
		current_value = CursorPatcher.db_count
		assert current_value == len(root.children) + 2
		assert CursorPatcher.db_count == current_value

		ContentmentCache().clear()
		CursorPatcher.reset()

		data = list(root.children)
		assert CursorPatcher.db_count == len(data) + 1

	def test_queryset_cache(self):
		q = Asset.objects
		assert len(ContentmentCache()) == 0
		q._populate_cache()
		assert len(ContentmentCache()) == Asset.objects.count()
		assert ContentmentCache().check_queryset(q)

		assert len(Asset.objects._search_in_cache({'order__gte': 4})) == 2
		assert len(Asset.objects._search_in_cache({'order__gt': 4})) == 1
		assert len(Asset.objects._search_in_cache({'order__lt': 2})) == 8
		assert len(Asset.objects._search_in_cache({'order__lte': 2})) == 11

		root = Asset.objects.get(path='/careers.illicohodes.com')
		db_count = CursorPatcher.db_count
		q_count = len(ContentmentCache().queries)
		Asset.objects.get(pk=root.pk)
		assert CursorPatcher.db_count == db_count
		assert len(ContentmentCache().queries) == q_count

		child1 = Asset.objects.get(path='/careers.illicohodes.com/theme')
		child2 = Asset.objects.get(path='/careers.illicohodes.com/theme/part')

		assert len(Asset.objects._search_in_cache({'parents__in': [child1]})) == 7
		assert len(Asset.objects._search_in_cache({'parents__in': [child2]})) == 3
		assert len(Asset.objects._search_in_cache({'parents__in': [child1, child2]})) == 7
		assert len(Asset.objects._search_in_cache({'parent__in': [child1, child2]})) == 7
		assert len(Asset.objects._search_in_cache({'parent__in': [child1]})) == 4
		assert len(Asset.objects._search_in_cache({'parent__in': [child2]})) == 3

	def test_ordering(self, root):
		qs = root.children
		assert [e.name for e in qs.order_by('-order')] == ['colophon', 'privacy', 'terms', 'careers', 'job', 'theme']
		assert [e.name for e in qs.order_by('parent', 'name')] == ['careers', 'colophon', 'job', 'privacy', 'terms', 'theme']

	def test_update(self, root):
		initial = [e.order for e in root.children]
		root.children.update(inc__order=2)
		assert [ContentmentCache()[str(e.pk)].order for e in root.children] == [i + 2 for i in initial]
		Asset.objects(pk=root.pk).update(set__name='TestName')
		assert ContentmentCache()[str(root.pk)].name == 'TestName'
		assert root._collection.find_one({'_id': root.pk})[root._fields['name'].db_field] == 'TestName'

		parent = Asset.objects.get(path='/careers.illicohodes.com/theme')

		data = list(parent.children)
		parent.children.update(pull_all__parents=[root, parent])
		assert all((parent not in e.parents and root not in e.parents) for e in data)

	def test_update_one(self, root):
		initial = [e.order for e in root.children]
		initial[0] += 2
		root.children.update_one(inc__order=2)
		assert [ContentmentCache()[str(e.pk)].order for e in root.children] == sorted(initial)

	def test_delete(self, root):
		child = Asset.objects.get(path='/careers.illicohodes.com/theme')
		Asset.objects(pk=child.pk).delete()
		assert child not in root.children
		assert child not in ContentmentCache()
