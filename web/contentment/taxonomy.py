# encoding: utf-8

from itertools import chain

from bson import ObjectId
from mongoengine.base.metaclasses import TopLevelDocumentMetaclass
from mongoengine import QuerySet, Q, DictField, MapField
from mongoengine import Document, ListField, StringField, IntField, ObjectIdField
from mongoengine import ReferenceField as BaseReferenceField
from mongoengine.connection import get_db
from mongoengine.signals import pre_delete
from mongoengine.common import _import_class
from mongoengine.base.fields import ComplexBaseField
from mongoengine.base.datastructures import EmbeddedDocumentList, BaseList, BaseDict

from web.ext.contentment import ContentmentCache

from .util.model import signal

log = __import__('logging').getLogger(__name__)


class CacheableQuerySet(QuerySet):
	def __init__(self, document, collection):
		super(CacheableQuerySet, self).__init__(document, collection)
		self._cache_iterator = None
		self._count = collection.count()
		self._db_will_be_queried = True

	@property
	def _db_query_required(self):
		return len(ContentmentCache()) < self._count

	@staticmethod
	def _search_in_cache(query):
		def exact_handler(entry, field, value):
			return getattr(entry, field) == value

		def ne_handler(entry, field, value):
			return not exact_handler(entry, field, value)

		def in_handler(entry, field, value):
			if isinstance(entry._fields.get(field), ListField):
				return set(getattr(entry, field, [])) & (set(value) if value else {})
			return getattr(entry, field, []) in (value if value is not None else [])

		def nin_handler(entry, field, value):
			return not in_handler(entry, field, value)

		def gt_handler(entry, field, value):
			return getattr(entry, field) > value

		def gte_handler(entry, field, value):
			return getattr(entry, field) >= value

		def lt_handler(entry, field, value):
			return getattr(entry, field) < value

		def lte_handler(entry, field, value):
			return getattr(entry, field) <= value

		HANDLERS = {
			'exact': exact_handler,
			'ne': ne_handler,
			'in': in_handler,
			'nin': nin_handler,
			'gt': gt_handler,
			'gte': gte_handler,
			'lt': lt_handler,
			'lte': lte_handler,
		}

		result = list(ContentmentCache().values())
		for part, value in query.items():
			field = part.rsplit('__', 1)
			if len(field) > 1:
				field, op = field
			else:
				field = field[0]
				op = 'exact'
			print(field, op, value)
			result = [entry for entry in result if HANDLERS.get(op)(entry, field, value)]
		if result:
			print('From cache: %s' % len(result))
		return result

	def _super_populate_cache(self):
		"""
		Populates the result cache with ``ITER_CHUNK_SIZE`` more entries
		(until the cursor is exhausted).
		"""
		from mongoengine.queryset.queryset import ITER_CHUNK_SIZE

		if self._result_cache is None:
			self._result_cache = []
		if self._has_more:
			try:
				for i in range(ITER_CHUNK_SIZE):
					entry = super(CacheableQuerySet, self).__next__()
					if entry not in self._result_cache:
						self._result_cache.append(entry)
			except StopIteration:
				self._has_more = False

	def __call__(self, q_obj=None, class_check=True, read_preference=None, **query):
		if not self._db_query_required:
			self._db_will_be_queried = False
		return super(CacheableQuerySet, self).__call__(q_obj, class_check, read_preference, **query)

	def clone(self):
		result = super(CacheableQuerySet, self).clone()
		result._db_will_be_queried = self._db_will_be_queried
		return result

	def _populate_cache(self):
		if self._db_query_required:
			self._super_populate_cache()
			for entry in self:
				if isinstance(entry, Document):
					ContentmentCache().store(entry)
		else:
			self._result_cache = self._search_in_cache(self._query_obj.query)
			if self._scalar:
				self._result_cache = [self._get_scalar(doc) for doc in self._result_cache]
			self._has_more = False

	def next(self):
		if self._db_will_be_queried:
			result = super(CacheableQuerySet, self).__next__()
			if isinstance(result, Document):
				ContentmentCache().store(result)
			return result

		if self._cache_iterator is None:
			self._result_cache = self._search_in_cache(self._query_obj.query)

			self._cache_iterator = iter(self._result_cache)

		result = next(self._cache_iterator)
		return result

	__next__ = next

	def order_by(self, *keys):

		def _get_order_tuple(doc):
			from mongoengine.fields import IntField, FloatField, DecimalField, StringField, BooleanField, ReferenceField
			from decimal import Decimal
			from bson.objectid import ObjectId

			DEFAULTS = {
				IntField: 0,
				FloatField: 0.0,
				DecimalField: Decimal(0),
				StringField: '',
				BooleanField: False,
				ReferenceField: ObjectId('0' * 24),
			}

			result = []

			for key in keys:
				attr = getattr(doc, key)
				if isinstance(attr, Document):
					result.append(attr.pk)
				elif attr is None:
					result.append(DEFAULTS.get(type(doc._fields[key])))
				else:
					result.append(attr)

			return tuple(result)

		if not self._db_will_be_queried:
			print('Sorted cache')
			reverse = False
			keys = list(keys)
			for i in range(len(keys)):
				key = keys[i]
				if key.startswith('-'):
					reverse = True
					keys[i] = key[1:]
			result = list(self)
			result.sort(key=_get_order_tuple, reverse=reverse)
			queryset = self.clone()
			queryset._result_cache = result
			queryset._has_more = self._has_more
			self.rewind()
			return queryset

		else:
			return super(CacheableQuerySet, self).order_by(*keys)

	def _update_in_cache(self, docs, update):
		def inc_handler(entry, field, value):
			setattr(entry, field, getattr(entry, field) + value)

		def set_handler(entry, field, value):
			setattr(entry, field, value)

		def pull_all_handler(entry, field, value):
			setattr(entry, field, [item for item in getattr(entry, field) if item not in set(value)])

		if '__raw__' in update:
			[ContentmentCache().store(doc.reload()) for doc in docs]
			return

		HANDLERS = {
			'inc': inc_handler,
			'set': set_handler,
			'pull_all': pull_all_handler,
		}

		from functools import partial

		updaters = []

		for part, value in update.items():
			part = part.split('__', 1)
			if len(part) < 2:
				continue
			op, field = part
			updaters.append(partial(HANDLERS.get(op), field=field, value=value))

		for doc in docs:
			[updater(doc) for updater in updaters]

	def update(self, upsert=False, multi=True, write_concern=None, full_result=False, **update):
		self._update_in_cache(self._search_in_cache(self._query_obj.query), update)
		return super(CacheableQuerySet, self).update(upsert, multi, write_concern, full_result, **update)

	def update_one(self, upsert=False, write_concern=None, **update):
		self._update_in_cache([self._search_in_cache(self._query_obj.query)[0]], update)
		return super(CacheableQuerySet, self).update_one(upsert, write_concern, **update)

	def delete(self, write_concern=None, _from_doc_delete=False, cascade_refs=None):
		for doc in self._search_in_cache(self._query_obj.query):
			ContentmentCache().remove(doc)
		return super(CacheableQuerySet, self).delete(write_concern, _from_doc_delete, cascade_refs)


@signal(pre_delete)
def remove_children(sender, document, **kw):
	document.empty()


from mongoengine import EmbeddedDocument
from bson import DBRef, SON
from mongoengine.dereference import DeReference
from mongoengine.base import get_document, TopLevelDocumentMetaclass


def get_from_cache(getter, model_id):
	cache = ContentmentCache()

	try:
		entry = cache[model_id]
	except KeyError:
		entry = getter()
		cache[model_id] = entry

	return entry


class CustomDereference(DeReference):
	def _find_references(self, items, depth=0, finded_ids=None):
		"""
		Recursively finds all db references to be dereferenced

		:param items: The iterable (dict, list, queryset)
		:param depth: The current depth of recursion
		"""

		reference_map = {}
		if not items or depth >= self.max_depth:
			return reference_map

		# Determine the iterator to use
		if not hasattr(items, 'items'):
			iterator = enumerate(items)
		else:
			iterator = iter(items.items())

		# Recursively find dbreferences
		depth += 1
		processed_ids = finded_ids if finded_ids is not None else set()
		for k, item in iterator:
			if isinstance(item, (Document, EmbeddedDocument)):
				for field_name, field in item._fields.items():
					v = item._data.get(field_name, None)
					if (v and getattr(field, 'document_type', object) is Taxonomy and
							isinstance(v, Taxonomy) and
							v.id not in processed_ids):
						processed_ids.add(v.id)
						reference_map.setdefault(type(v), set()).add(v.id)
					elif isinstance(v, DBRef) and v.id not in processed_ids:
						processed_ids.add(v.id)
						try:
							reference_map.setdefault(get_document(v.cls), set()).add(v.id)
						except AttributeError:
							reference_map.setdefault(field.document_type, set()).add(v.id)
					elif isinstance(v, (dict, SON)) and '_ref' in v and v['_ref'].id not in processed_ids:
						processed_ids.add(v['_ref'].id)
						reference_map.setdefault(get_document(v['_cls']), set()).add(v['_ref'].id)
					elif isinstance(v, (dict, list, tuple)) and depth <= self.max_depth:
						field_cls = getattr(getattr(field, 'field', None), 'document_type', None)
						references = self._find_references(v, depth, processed_ids)
						for key, refs in references.items():
							if isinstance(field_cls, (Document, TopLevelDocumentMetaclass)) and key is Taxonomy:
								key = field_cls
							reference_map.setdefault(key, set()).update(refs)
			elif isinstance(item, DBRef) and item.id not in processed_ids:
				processed_ids.add(item.id)
				reference_map.setdefault(item.collection, set()).add(item.id)
			elif isinstance(item, (dict, SON)) and '_ref' in item and item['_ref'].id not in processed_ids:
				processed_ids.add(item['_ref'].id)
				reference_map.setdefault(get_document(item['_cls']), set()).add(item['_ref'].id)
			elif isinstance(item, (dict, list, tuple)) and depth - 1 <= self.max_depth:
				references = self._find_references(item, depth - 1, processed_ids)
				for key, refs in references.items():
					reference_map.setdefault(key, set()).update(refs)

		return reference_map

	def _fetch_objects(self, doc_type=None):
		"""Fetch all references and convert to their document objects
		"""
		cache = ContentmentCache()

		object_map = {}
		for collection, dbrefs in self.reference_map.items():
			cached = set(dbref for dbref in dbrefs if str(dbref) in (cache or {}))
			if hasattr(collection, 'objects'):  # We have a document class for the refs
				col_name = collection._get_collection_name()
				refs = [dbref for dbref in dbrefs
						if ((col_name, dbref) not in object_map and dbref not in cached)]
				references = collection.objects.in_bulk(refs)
				for key, doc in references.items():
					object_map[(col_name, key)] = doc
					if cache is not None:
						cache[str(key)] = doc
				for dbref in cached:
					object_map[(col_name, dbref)] = cache[str(dbref)]
			else:  # Generic reference: use the refs data to convert to document
				if isinstance(doc_type, (ListField, DictField, MapField)):
					continue

				refs = [dbref for dbref in dbrefs
						if (collection, dbref) not in object_map and dbref not in cached]

				if doc_type:
					references = doc_type._get_db()[collection].find({'_id': {'$in': refs}})
					for ref in references:
						doc = doc_type._from_son(ref)
						object_map[(collection, doc.id)] = doc
						if cache is not None:
							cache[str(doc.id)] = doc
				else:
					references = get_db()[collection].find({'_id': {'$in': refs}})
					for ref in references:
						if '_cls' in ref:
							doc = get_document(ref["_cls"])._from_son(ref)
						elif doc_type is None:
							doc = get_document(
								''.join(x.capitalize()
										for x in collection.split('_')))._from_son(ref)
						else:
							doc = doc_type._from_son(ref)
						object_map[(collection, doc.id)] = doc
						if cache is not None:
							cache[str(doc.id)] = doc
				for dbref in cached:
					object_map[(collection, dbref)] = cache[str(dbref)]
		return object_map


class ReferenceField(BaseReferenceField):
	def __get__(self, instance, owner):
		"""Descriptor to allow lazy dereferencing.
		"""
		if instance is None:
			# Document class being used rather than a document object
			return self

		# Get value from document instance if available
		value = instance._data.get(self.name)
		self._auto_dereference = instance._fields[self.name]._auto_dereference
		# Dereference DBRefs
		if self._auto_dereference and isinstance(value, DBRef):
			if hasattr(value, 'cls'):
				# Dereference using the class type specified in the reference
				cls = get_document(value.cls)
			else:
				cls = self.document_type

			def get_model():
				data = cls._get_db().dereference(value)
				if data is not None:
					return cls._from_son(data)

			instance._data[self.name] = get_from_cache(get_model, str(getattr(value, 'id')))

		return super(ReferenceField, self).__get__(instance, owner)


def _get_field_from_db(asset, field):
	pk = asset._fields.get(asset._meta.get('id_field'))
	if pk is None:
		return None

	field_obj = asset._fields.get(field)
	if field_obj is None:
		raise ValueError('Invalid field name: %s' % field)

	return asset._collection.find_one({pk.db_field: asset.pk})[field_obj.db_field]


class TaxonomyQuerySet(CacheableQuerySet):
	def __init__(self, document, collection, _rewrite_initial=False):
		super(TaxonomyQuerySet, self).__init__(document, collection)
		self.__dereference = None
		if _rewrite_initial:
			self._initial_query = {'_cls': {'$in': Taxonomy._subclasses}}

	# @property
	# def _cursor(self):
	# 	print("Cursor")
	# 	return super(TaxonomyQuerySet, self)._cursor

	def process_assets(self, assets, sort_fn=None):
		if sort_fn is None:
			sort_fn = lambda it: (str(it.parent.id) if it.parent else '', it.order)
		result = sorted(assets, key=sort_fn)
		qs = self.base_query(pk__in=[asset.pk for asset in result])
		qs._result_cache = result
		qs._has_more = False
		return qs

	def _to_qs(self, data):
		if isinstance(data, QuerySet):
			return data
		return self.base_query(pk__in=[asset.pk for asset in data])

	def _get_from_db(self, assets, fn, exclude=None, order=('parent', 'order')):
		from operator import __or__
		from functools import reduce

		query = [fn(asset) for asset in assets]

		# for asset in assets:#.no_dereference():
		# 	query.append(Q(parent=asset.parent, pk__ne=asset.pk))

		if not query:  # TODO: Armour everywhere.
			return []

		query = reduce(__or__, query)
		if exclude:
			query &= Q(id__nin=exclude)

		return self.base_query(query).order_by(*(order if isinstance(order, (list, tuple)) else [order]))

	@property
	def _dereference(self):
		if not self.__dereference:
			self.__dereference = CustomDereference()
		return self.__dereference

	@property
	def base_query(self):
		return TaxonomyQuerySet(Taxonomy, self._collection)

	# Quick Lookup

	def named(self, name):
		return self.filter(name=name)

	def nearest(self, path):
		if hasattr(path, 'split'):
			path = path.split('/')

		# Remove leading empty elements.
		while path and not path[0]:
			del path[0]

		# Remove trailing empty elements.
		while path and not path[-1]:
			del path[-1]

		# Determine the full list of possible paths.
		paths = [('/' + '/'.join(path[:i])) for i in range(len(path) + 1)]

		log.debug("Searching for element nearest: /" + '/'.join(path), extra=dict(search=paths))

		# Find the deepest (completely or partially) matching asset.
		return self.clone().filter(path__in=paths).order_by('-path').first()

	# Basic Management

	def empty(self, *q_objs, **query):
		"""Delete all descendants of the currently selected assets.

		Warning: If run on all assets this will only leave the root element intact. It would also be expensive.
		"""

		# import ipdb; ipdb.set_trace()

		parents = self.clone().filter(*q_objs, **query)

		# Optimization note: this doesn't need to worry about normalizing paths, thus the _from_doc_delete.
		# TODO: Handle potential exception: signal handlers may preemptively delete included records. That's perfectly ok!
		self.base_query(parents__in=parents).delete(write_concern=None, _from_doc_delete=True)  # TODO: write_concern

		# Returns original QuerySet, as it'll need to re-query to check if any included results survive.
		return self

	def insert(self, index, child):
		"""Add an asset, specified by the parameter, as a child of this asset."""

		parent = self.clone().first()

		log.info("Inserting asset.", extra=dict(asset=parent.id, index=index, child=getattr(child, 'id', child)))

		# Detach the new child (and thus it's own child nodes).
		child = (self.base_query.get(id=child) if isinstance(child, ObjectId) else child).detach(False)

		if index < 0:
			_max = self.base_query(parent=parent).order_by('-order').scalar('order').first()
			index = 0 if _max is None else (_max + 1)

		self.base_query(parent=parent, order__gte=index).update(inc__order=1)

		log.debug("before", extra=dict(data=repr(child._data)))

		child.order = index

		child.path = parent.path + '/' + child.name
		child.parent = parent
		child.parents = list(parent.parents)
		child.parents.append(parent)

		log.debug("after", extra=dict(data=repr(child._data)))

		child = child.save()

		print("Child contents:", child.contents)
		child.contents.update(__raw__=
			{'$push': {
				't_a': {
					'$each': [i.to_dbref() for i in child.parents],
					'$position': 0
				}
			}})

		# import ipdb; ipdb.set_trace()

		# child.contents.update(push__ancestors={'$each': ancestors, '$position': 0})  # Unimplemented.

		parent._normpath(child.id)

		return self

	def detach(self, path=True):
		"""Detach this asset from its current taxonomy."""

		obj = self.clone().first()

		if obj.path in (None, '', obj.name):
			return obj

		log.warn("Detaching from taxonomy." + "\n\t" + __import__('json').dumps(dict(asset=repr(obj), path=path)))

		self.nextAll.update(inc__order=-1)

		self.contents.update(pull_all__parents=obj.parents)

		obj.order = None
		obj.path = obj.name
		obj.parent = None
		# We can't use `del obj.parents[:]` because of some unintentional mongoengine behaviour
		obj.parents.clear()

		if path:
			obj._normpath(obj.id)

		obj.save()

		return self

	def append(self, child):
		"""Insert an asset, specified by the parameter, as a child of this asset."""
		return self.insert(-1, child)

	def prepend(self, child):
		return self.insert(0, child)

	def after(self, sibling):
		"""Insert an asset, specified by the parameter, after this asset."""
		obj = self.clone().first()
		obj.parent.insert(obj.order + 1, sibling)
		return self

	def before(self, sibling):
		"""Insert an asset, specified by the parameter, before this asset."""
		obj = self.clone().first()
		obj.parent.insert(obj.order, sibling)
		return self

	def replace(self, target):
		"""Replace an asset, specified by the parameter, with this asset."""

		target = self.clone().get(id=target) if isinstance(target, ObjectId) else target
		obj = self.clone().first()

		obj.name = target.name
		obj.parent = target.parent
		obj.parents = target.parents
		obj.path = target.path
		obj.order = target.order

		ContentmentCache().remove(target)
		target.delete()
		obj.save()

		return self

	def replaceWith(self, source):
		"""Replace this asset with an asset specified by the parameter."""

		source = self.clone().get(id=source) if isinstance(source, ObjectId) else source
		obj = self.clone().first()

		source.name = obj.name
		source.parent = obj.parent
		source.parents = obj.parents
		source.path = obj.path
		source.order = obj.order

		ContentmentCache().remove(obj)
		obj.delete()
		source.save()

		return self

	def clone_assets(self):
		clones = self.clone()
		for clone in clones:
			del clone.id

		return clones

	# Pivoted Manipulation
	# These are actually implemented elsewhere.

	def appendTo(self, parent):
		"""Insert this asset as a child of the asset specified by the parameter."""
		return self.base_query(pk=getattr(parent, 'pk', parent)).append(self.clone().first())

	def prependTo(self, parent):
		"""Insert this asset as the left-most child of the asset specified by the parameter."""
		return self.base_query(pk=getattr(parent, 'pk', parent)).prepend(self.clone().first())

	def insertBefore(self, sibling):
		"""Insert this asset as the left-hand sibling of the asset specified by the parameter."""
		return self.base_query(pk=getattr(sibling, 'pk', sibling)).before(self.clone().first())

	def insertAfter(self, sibling):
		"""Insert this asset as the right-hand child of the asset specified by the parameter."""
		return self.base_query(pk=getattr(sibling, 'pk', sibling)).after(self.clone().first())

	# Traversal

	@property
	def children(self):
		"""Yield all direct children of this asset."""

		return self.base_query(parent__in=self.clone()).order_by('parent', 'order')

	@property
	def contents(self):
		"""Yield all descendants of this asset."""

		return self.base_query(parents__in=self.clone().all()).order_by('parent', 'order')

	@property
	def siblings(self):
		"""All siblings of the currently selected assets, not including these assets."""
		assets = self.clone()
		filter_fn = lambda asset: Q(parent=asset.parent, pk__ne=asset.pk)

		return self._get_from_db(assets, filter_fn)

	def _next_prev(self, filter_fn):
		"""The sibling immediately following this asset."""
		assets = self.clone()

		return self._get_from_db(assets, filter_fn, order='path').first()

	@property
	def next(self):
		return self._next_prev(lambda asset: Q(parent=asset.parent, order=asset.order + 1))

	@property
	def nextAll(self):
		"""All siblings following this asset."""
		assets = self.clone()
		filter_fn = lambda asset: Q(parent=asset.parent, order__gt=asset.order, pk__ne=asset.pk)

		return self._get_from_db(assets, filter_fn)

	@property
	def prev(self):
		"""The sibling immediately preceeding this asset."""
		return self._next_prev(lambda asset: Q(parent=asset.parent, order=asset.order - 1))

	@property
	def prevAll(self):
		"""All siblings preceeding the selected assets."""
		assets = self.clone()
		filter_fn = lambda asset: Q(parent=asset.parent, order__lt=asset.order, pk__ne=asset.pk)

		return self._get_from_db(assets, filter_fn)

	def contains(self, other):
		"""The asset, specified by the parameter, is a descendant of any of the selected assets."""

		if __debug__:  # Can be optimized away (-O) in production.
			from web.component.asset import Asset
			assert isinstance(other, Asset) or isinstance(other, ObjectId), "Argument must be Asset or ObjectId instance."

		parents = self.clone().scalar('id').no_dereference()
		return bool(self.base_query(pk=getattr(other, 'pk', other), parents__in=parents).count())

	def extend(self, *others):
		"""Merge the contents of another asset or assets, specified by positional parameters, with this one."""
		obj = self.clone().first()
		for other in others:
			for child in other.children:
				obj.insert(-1, child)
		return self

	# def get(self, *q_objs, **query):
	# 	model_id = str(query['id'])
	# 	getter = lambda: super(TaxonomyQuerySet, self).get(*q_objs, **query)
	# 	return get_from_cache(getter, model_id)


class CustomDereferenceMixin(ComplexBaseField):
	def __get__(self, instance, owner):
		"""Descriptor to automatically dereference references.
		"""
		if instance is None:
			# Document class being used rather than a document object
			return self

		ReferenceField = _import_class('ReferenceField')
		GenericReferenceField = _import_class('GenericReferenceField')
		EmbeddedDocumentListField = _import_class('EmbeddedDocumentListField')
		dereference = (self._auto_dereference and
					   (self.field is None or isinstance(self.field,
														 (GenericReferenceField, ReferenceField))))

		_dereference = CustomDereference()

		self._auto_dereference = instance._fields[self.name]._auto_dereference
		if instance._initialised and dereference and instance._data.get(self.name):
			instance._data[self.name] = _dereference(
				instance._data.get(self.name), max_depth=1, instance=instance,
				name=self.name
			)

		value = super(ComplexBaseField, self).__get__(instance, owner)

		# Convert lists / values so we can watch for any changes on them
		if isinstance(value, (list, tuple)):
			if (issubclass(type(self), EmbeddedDocumentListField) and
					not isinstance(value, EmbeddedDocumentList)):
				value = EmbeddedDocumentList(value, instance, self.name)
			elif not isinstance(value, BaseList):
				value = BaseList(value, instance, self.name)
			instance._data[self.name] = value
		elif isinstance(value, dict) and not isinstance(value, BaseDict):
			value = BaseDict(value, instance, self.name)
			instance._data[self.name] = value

		if (self._auto_dereference and instance._initialised and
				isinstance(value, (BaseList, BaseDict)) and
				not value._dereferenced):
			value = _dereference(
				value, max_depth=1, instance=instance, name=self.name
			)
			value._dereferenced = True
			instance._data[self.name] = value

		return value


class CustomListField(CustomDereferenceMixin, ListField):
	pass


class CachedDocumentMetaclass(TopLevelDocumentMetaclass):
	def __call__(cls, *args, **values):
		cache = ContentmentCache()

		key = values.get(cls._meta.get('id_field'))
		key = str(key) or None

		if key is None or key not in cache:
			instance = super().__call__(*args, **values)
			if instance.pk:
				cache.store(instance)
			return instance

		return cache[key]


class Taxonomy(Document, metaclass=CachedDocumentMetaclass):
	meta = dict(
		id_field = 'id',
		ordering = ['order'],
		allow_inheritance = True,
		abstract = True,
		queryset_class = TaxonomyQuerySet,
	)

	parent = ReferenceField(
			'self',
			db_field = 't_p',
			export=False
		)
	parents = CustomListField(ReferenceField(
			'self',
		), db_field='t_a', export=False)

	id = ObjectIdField(db_field='_id', primary_key=True, default=ObjectId)
	name = StringField(db_field='n', export=True, simple=True)
	path = StringField(db_field='t_P', unique=True, export=True, simple=True)
	order = IntField(db_field='t_o', default=0, export=True, simple=True)

	# def __new__(cls, *args, **values):
	# 	cache = ContentmentCache()
	#
	# 	key = values.get(cls._meta.get('id_field'))
	# 	key = str(key) or None
	#
	# 	if key is None or key not in cache:
	# 		instance = super(Taxonomy, cls).__new__(cls)
	# 		instance.__init__(*args, **values)
	# 		if instance.pk:
	# 			print('Added to cache')
	# 			cache.store(instance)
	# 		return instance
	#
	# 	print('From cache')
	# 	return cache[key]

	def __init__(self, *a, **kw):
		super(Taxonomy, self).__init__(*a, **kw)

	def __repr__(self):
		return "{0.__class__.__name__} ({0.name}, {0.path})".format(self)

	def save(self, *args, **kwargs):
		result = super(Taxonomy, self).save(*args, **kwargs)
		ContentmentCache().store(result)
		return result

	@property
	def _qs(self):
		"""
		Returns the queryset to use for updating / reloading / deletions
		"""
		if not hasattr(self, '__objects'):
			self.__objects = self.tqs
		return self.__objects

	@property
	def tqs(self):
		return TaxonomyQuerySet(self.__class__, self._get_collection(), _rewrite_initial=True)

	def tree(self, indent=''):
		print(indent, repr(self), sep='')

		for child in self.children:
			child.tree(indent + '    ')

	def _normpath(self, parent):
		for child in self._get_collection().find({'t_a._id': parent}, {'n': 1, 't_a.n': 1}).sort('t_P'):
			self.tqs(id=child['_id']).update_one(set__path='/' + '/'.join(chain((i['n'] for i in child['t_a']), [child['n']])))

	def empty(self):
		self.tqs(id=self.id).empty()

	def detach(self, path=True):
		"""Detach this asset from its current taxonomy."""
		self.tqs(id=self.id).detach(path)
		return self.reload()

	def insert(self, index, child):
		"""Add an asset, specified by the parameter, as a child of this asset."""
		self.tqs(id=self.id).insert(index, child)
		return self.reload()

	def append(self, child):
		"""Insert an asset, specified by the parameter, as a child of this asset."""
		self.tqs(id=self.id).append(child)
		return self

	def prepend(self, child):
		self.tqs(id=self.id).prepend(child)
		return self

	def after(self, sibling):
		"""Insert an asset, specified by the parameter, after this asset."""
		self.tqs(id=self.id).after(sibling)
		return self

	def before(self, sibling):
		"""Insert an asset, specified by the parameter, before this asset."""
		self.tqs(id=self.id).before(sibling)
		return self

	def replace(self, target):
		"""Replace an asset, specified by the parameter, with this asset."""
		self.tqs(id=self.id).replace(target)
		return self

	def replaceWith(self, source):
		"""Replace this asset with an asset specified by the parameter."""
		self.tqs(id=self.id).replaceWith(source)
		return source

	def clone(self):
		return self.tqs(id=self.id).clone_assets()[0]

	def appendTo(self, parent):
		"""Insert this asset as a child of the asset specified by the parameter."""
		self.tqs(id=self.id).appendTo(parent)
		return self

	def prependTo(self, parent):
		"""Insert this asset as the left-most child of the asset specified by the parameter."""
		self.tqs(id=self.id).prependTo(parent)
		return self

	def insertBefore(self, sibling):
		"""Insert this asset as the left-hand sibling of the asset specified by the parameter."""
		self.tqs(id=self.id).insertBefore(sibling)
		return self

	def insertAfter(self, sibling):
		"""Insert his asset as the right-hand child of the asset specified by the parameter."""
		self.tqs(id=self.id).insertAfter(sibling)
		return self

	@property
	def children(self):
		"""Yield all direct children of this asset."""
		return self.tqs(id=self.id).children

	@property
	def contents(self):
		"""Yield all descendants of this asset."""
		return self.tqs(id=self.id).contents

	@property
	def siblings(self):
		"""All siblings of this asset, not including this asset."""
		return self.tqs(id=self.id).siblings

	@property
	def next(self):
		"""The sibling immediately following this asset."""
		return self.tqs(id=self.id).next

	@property
	def nextAll(self):
		"""All siblings following this asset."""
		return self.tqs(id=self.id).nextAll

	@property
	def prev(self):
		"""The sibling immediately preceeding this asset."""
		return self.tqs(id=self.id).prev

	@property
	def prevAll(self):
		"""All siblings preceeding this asset."""
		return self.tqs(id=self.id).prevAll

	def contains(self, other):
		"""The asset, specified by the parameter, is a descendant of this asset."""
		return self.tqs(id=self.id).contains(other)

	def extend(self, *others):
		"""Merge the contents of another asset or assets, specified by positional parameters, with this one."""
		self.tqs(id=self.id).extend(*others)
		return self
