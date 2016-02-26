# encoding: utf-8

import pytest

from test.conftest import TModel


@pytest.mark.usefixtures('connection')
class TestTaxonomy:
	def test_siblings(self, root):
		assert len(root.children[1].siblings) == 5

	def test_next(self, root):
		subroot = root.children[0].children[1]
		nexts = list(root.children[0].children[2:4])
		assert len(subroot.nextAll) == 2
		assert list(subroot.nextAll) == nexts
		c = subroot
		count = 0
		while True:
			c = c.next
			if c is None:
				break
			count += 1
		assert count == 2

	def test_prev(self, root):
		subroot = root.children[0].children[2]
		prevs = list(root.children[0].children[0:2])
		assert len(subroot.prevAll) == 2
		assert list(subroot.prevAll) == prevs
		c = subroot
		count = 0
		while True:
			c = c.prev
			if c is None:
				break
			count += 1
		assert count == 2

	def test_insert_detach(self, models):
		assert len(models.parent.children) == 0
		assert models.child1.parents == []

		models.parent.insert(0, models.child1)
		models.parent.insert(1, models.child2)
		assert list(models.parent.children) == [models.child1, models.child2]

		models.child2.detach()
		assert list(models.parent.children) == [models.child1]

		models.parent.insert(0, models.child2)
		assert list(models.parent.children) == [models.child2, models.child1]

		models.child1.append(models.child2)
		assert list(models.parent.children) == [models.child1] and list(models.child1.children) == [models.child2]
		assert list(models.parent.contents) == [models.child1, models.child2]

		models.child1.empty()
		assert list(models.parent.contents) == [models.child1]
		assert TModel.objects.count() == 2

	def test_append_prepend(self, models):
		models.parent.append(models.child1)
		assert list(models.parent.children) == [models.child1]
		models.parent.append(models.child2)
		assert list(models.parent.children) == [models.child1, models.child2]

		models.parent.prepend(models.child2)
		assert list(models.parent.children) == [models.child2, models.child1]

	def test_after_before(self, models):
		models.parent.append(models.child1)
		models.child1.after(models.child2)
		assert list(models.parent.children) == [models.child1, models.child2]

		models.child1.before(models.child2)
		assert list(models.parent.children) == [models.child2, models.child1]

	def test_pivoted(self, models):
		models.child1.appendTo(models.parent)
		assert list(models.parent.children) == [models.child1]
		models.child2.prependTo(models.parent)
		assert list(models.parent.children) == [models.child2, models.child1]

		models.child1.insertBefore(models.child2)
		assert list(models.parent.children) == [models.child1, models.child2]
		models.child1.insertAfter(models.child2)
		assert list(models.parent.children) == [models.child2, models.child1]

	def test_extend(self, models):
		parent2 = TModel.objects.create(name='par2', path='/P2')
		models.child1.appendTo(models.parent)
		models.child2.appendTo(parent2)

		models.parent.extend(parent2)
		assert list(models.parent.children) == [models.child1, models.child2]

	def test_replace(self, models):
		replacer = TModel.objects.create(name='replacer', path='r')
		models.child1.appendTo(models.parent)
		models.child1.reload(-1)
		assert TModel.objects.count() == 4
		replacer.replace(models.child1)
		assert TModel.objects.count() == 3
		assert list(models.parent.children) == [replacer.reload(-1)]

		models.child2.replaceWith(replacer)
		replacer.reload(-1)
		assert len(models.parent.children) == 0
		assert replacer.name == 'child2'
