__constant char message[] = "Hello, World!\0";

__kernel void hello(__global char* string, __local int *index)
{
	__private int id = get_local_id(0);
	string[id+*index] = message[id+*index];
	*index += 1;
	//printf("What's my index? It's g%d:l%d:f%d It's %d\n", get_global_id(0), get_local_id(0), get_local_id(1), *index);
}

