#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <functional>
#include <iostream>
#include <vector>

template<class Key, class Data>
struct HashEntry
{
	Key key;
	Data data;
	int next;
};

// a useful core hash function
inline unsigned int hashk(unsigned int k)
{ return k*2654435769u; }

// default hash function object
struct DefaultHashFunction
{
	template<typename Key>
	unsigned int operator() (const Key &k) const { return hashk(k); }
};

struct equalto
{
	template<typename T>
	bool operator() (const T &a, const T &b) const { return a==b; }
};

template<typename Key, typename Data, class HashFunction=DefaultHashFunction, class KeyEqual=equalto>
struct HashTable
{
	unsigned int table_rank;
	unsigned int table_bits;
	std::vector<int> table;
	unsigned int num_entries;
	std::vector<HashEntry<Key, Data> > pool;
	int free_list;
	const HashFunction hash_function;
	const KeyEqual key_equal;

	explicit HashTable(unsigned int expected_size=64)
		: hash_function(HashFunction()), key_equal(KeyEqual())
	{ init(expected_size); }

	explicit HashTable(const HashFunction &hf, unsigned int expected_size=64)
		: hash_function(hf), key_equal(KeyEqual())
	{ init(expected_size); }

	void init(unsigned int expected_size)
	{
		unsigned int i;
		num_entries=0;
		table_rank=4;
		while(1u<<table_rank < expected_size)
			++table_rank;
		++table_rank; // give us some extra room
		table_bits=(1u<<table_rank)-1;
		table.resize(1u<<table_rank);
		for(i=0; i<table.size(); ++i)
			table[i]=-1; // empty list
		pool.resize(1u<<table_rank);
		free_list=0;
		for(unsigned int i=0; i<pool.size()-1; ++i)
			pool[i].next=i+1;
		pool[pool.size()-1].next=-1; // end of free list
	}

	void add(const Key &k, const Data &d)
	{
		if(free_list==-1)
			reserve(1u<<(table_rank+1));
		int i=free_list; // where we're going to put the new entry
		free_list=pool[i].next;
		unsigned int t=hash_function(k)&table_bits; // index into table
		pool[i].key=k;
		pool[i].data=d;
		pool[i].next=table[t]; // put the new entry at the start of table[t]'s list
		table[t]=i;
		++num_entries;
	}

	void delete_entry(const Key &k, const Data &d) // delete first entry that matches both key and data
	{
		unsigned int t=hash_function(k)&table_bits;
		int i=table[t], *p_i=&table[t];
		while(i!=-1){
			if(key_equal(k, pool[i].key) && d==pool[i].data){
				*p_i=pool[i].next; // make list skip over this entry
				pool[i].next=free_list; // and put it on the front of the free list
				free_list=i;
				return; // and we're done
			}
			p_i=&pool[i].next;
			i=*p_i;
		}
	}

	unsigned int size() const
	{ return num_entries; }

	void clear()
	{
		unsigned int i=0;
		num_entries=0;
		for(i=0; i<table.size(); ++i)
			table[i]=-1; // empty list
		free_list=0;
		for(i=0; i<pool.size()-1; ++i)
			pool[i].next=i+1;
		pool[pool.size()-1].next=-1;
	}

	void reserve(unsigned int expected_size)
	{
		if(expected_size<=pool.size())
			return;
		while(1u<<table_rank < expected_size)
			++table_rank;
		table_bits=(1u<<table_rank)-1;
		// increase room for new entries
		unsigned int old_size=(unsigned int)pool.size(), i;
		pool.resize(1u<<table_rank);
		for(i=old_size; i<pool.size()-1; ++i)
			pool[i].next=i+1;
		pool[i].next=free_list;
		free_list=old_size;
		// And finally need to redo table (rehash entries)
		old_size=(unsigned int)table.size();
		table.resize(1u<<table_rank);
		unsigned int t;
		for(t=old_size; t<table.size(); ++t)
			table[t]=-1; // initially make new lists empty
		int j, *p_j;
		for(t=0; t<old_size; ++t){
			j=table[t]; 
			p_j=&table[t];
			while(j!=-1){
				unsigned int new_t=hash_function(pool[j].key)&table_bits;
				if(new_t!=t){ // j doesn't belong in this list anymore?
					// delete from this list
					*p_j=pool[j].next;
					// add to correct list
					pool[j].next=table[new_t];
					table[new_t]=j;
				}else
					p_j=&(pool[j].next);
				j=*p_j;
			}
		}
	}

	bool has_entry(const Key &k) const
	{
		unsigned int t=hash_function(k)&table_bits;
		int i=table[t];
		while(i!=-1){
			if(key_equal(k, pool[i].key))
				return true;
			i=pool[i].next;
		}
		return false;
	}

	bool get_entry(const Key &k, Data &data_return) const
	{
		unsigned int t=hash_function(k)&table_bits;
		int i=table[t];
		while(i!=-1){
			if(key_equal(k, pool[i].key)){
				data_return=pool[i].data;
				return true;
			}
			i=pool[i].next;
		}
		return false;
	}

	void append_all_entries(const Key& k, std::vector<Data>& data_return) const
	{
		unsigned int t=hash_function(k)&table_bits;
		int i=table[t];
		while(i!=-1){
			if(key_equal(k, pool[i].key)) data_return.push_back(pool[i].data);
			i=pool[i].next;
		}
	}

	Data &operator() (const Key &k, const Data &missing_data)
	{
		unsigned int t=hash_function(k)&table_bits;
		int i=table[t];
		while(i!=-1){
			if(key_equal(k, pool[i].key))
				return pool[i].data;
			i=pool[i].next;
		}
		add(k, missing_data); // note - this could cause the table to be resized, and t made out-of-date
		return pool[table[hash_function(k)&table_bits]].data; // we know that add() puts it here!
	}

	const Data &operator() (const Key &k, const Data &missing_data) const
	{
		unsigned int t=hash_function(k)&table_bits;
		int i=table[t];
		while(i!=-1){
			if(key_equal(k, pool[i].key))
				return pool[i].data;
			i=pool[i].next;
		}
		return missing_data;
	}

	void output_statistics() const
	{
		std::vector<int> lengthcount(table.size());
		unsigned int t;
		int total=0;
		for(t=0; t<table.size(); ++t){
			int i=table[t], length=0;
			while(i!=-1){
				++length;
				i=pool[i].next;
			}
			++lengthcount[length];
			++total;
		}
		int subtotal=0;
		int maxlength=0;
		for(t=0; t<lengthcount.size() && t<10; ++t){
			subtotal+=lengthcount[t];
			if(lengthcount[t]>0){
				std::cout<<"length "<<t<<": "<<lengthcount[t]<<"   ("<<lengthcount[t]/(float)total*100.0<<"%)"<<std::endl;
				maxlength=t;
			}
		}
		std::cout<<"rest: "<<total-subtotal<<"   ("<<100.0*(1.0-subtotal/(float)total)<<"%)"<<std::endl;
		for(; t<lengthcount.size(); ++t)
			if(lengthcount[t]>0)
				maxlength=t;
		std::cout<<"longest list: "<<maxlength<<std::endl;
	}
};

#endif
