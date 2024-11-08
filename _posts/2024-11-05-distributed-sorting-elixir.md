---
layout: home
title:  "Distributed sorting with Elixir genservers"
date:  2024-04-01 
permalink: /distributed-sorting-elixir-genservers/
categories: elixir algorithm
image: distributed-sorting-elixir-genservers.png
---


# Introduction

In this post I am going to discuss how to implement a distributed sorting algorithm using Elixir genservers. The algorithm is based on the quicksort algorithm, which is a comparison sort algorithm that uses a divide-and-conquer strategy to sort an array. The algorithm is implemented in a distributed way, where each process is responsible for sorting a subset of the array. The processes communicate with each other to exchange information about the sorted subsets and to merge them into a final sorted array.


> :information_source: **Disclaimer**: [This is probably not the best use of genservers](https://hexdocs.pm/elixir/GenServer.html#module-when-not-to-use-a-genserver) and could have much easily been implemented using other Elixir abstractions such as tasks. However, since I wanted to learn how to use Genservers, I decided to implement the algorithm using them.

You can find the full code for this implementation in this [Github repository](https://github.com/andre-b-fernandes/distributed_sorting/tree/main).


![gb](/assets/img/posts/distributed-sorting-elixir-genservers/distributed_algorithm.png)

The algorithm works as follows:

- The main process receives a list to be sorted and the number of processes to be used.
- The main process creates a master process that will be responsible for coordinating the sorting process.
- The master process divides the array into `x` smaller subsets, where `x` is the result of the array being divided by a constant number `k`.
- The master process creates `y` genservers, each responsible for sorting a subset of the array.
- The master process sends the subsets to the genservers.
- Each genserver sorts its subset using the quicksort algorithm.
- Each genserver sends the sorted subset back to the main process.
- At each 2 received sorted arrays, the master process sends them to a genserver that merges them into a single sorted subset and returns it back to the master process.
- The process is repeated until there is only one sorted subset left with the same size of the original list.


# Elixir implementation

## Mix

The project was created via `mix` which is a build tool that provides tasks for creating, compiling, testing your application, managing its dependencies and much more.

In the project definition we define the application to come from our `DistributedSorting` module:

```elixir
def application do
    app = [
      extra_applications: [:logger]
      # mod: {DistributedSorting, []} # Only if not Mix.env() == :test
    ]

    if Mix.env() != :test do
      app ++ [mod: {DistributedSorting, []}]
    else
      app
    end
  end
```

Run the application as ``mix run -- --file yourfilepath.txt --workers 100``

## Application behaviour

This module uses the `Application` [behaviour](https://hexdocs.pm/elixir/1.12/Application.html) and is responsible for starting and stopping the application. It is also responsible for starting the main process that will coordinate the sorting process.

```elixir
defmodule DistributedSorting do
  use Application

  def read_file_to_sort(file_path) do
    {:ok, content} = File.read(file_path)

    content
    |> String.split("\n")
    |> Enum.map(&String.to_integer/1)
  end

  defp parse_args(args) do
    {parsed, _, _} = OptionParser.parse(args, strict: [file: :string, workers: :integer])
    input_file = parsed[:file]
    n_workers = parsed[:workers]
    {input_file, n_workers}
  end

  def start(_start_type, _start_args) do
    args = System.argv()
    {input_file, n_workers} = parse_args(args)
    array_to_sort = read_file_to_sort(input_file)

    {:ok, pid} = DistributedMaster.start_link({array_to_sort, n_workers})

    ref = Process.monitor(pid)

    receive do
      {:DOWN, ^ref, :process, _object, _reason} ->
        IO.puts("Master process terminated")
    end

    {:ok, pid}
  end
end
```

The module starts the application by reading a input file whose path is passed as an argument. The file contains a list of integers to be sorted in the following format:

```
99022
20655
-63097
88111
4412
3499
42423
91401
61189
-88446
37496
-71973
21591
-27418
-42662
95704
68402
-45611
-89018
(...)
```

## Genservers

The module then starts the main process `DistributedMaster` that will coordinate the sorting process.

```elixir
defmodule DistributedMaster do
  @moduledoc """
  Distributed Master module.

  This module implements a GenServer that starts worker processes and sends lists to be sorted.
  It initally calls the worker processes to sort the lists and then merges the sorted lists.
  It implements 1 different call message:
  - {:add_sorted_list, list} - Adds a sorted list to the list of sorted lists and merges two lists when two lists are sorted.

  Once it finishes it will write the sorted list to a file and terminate the worker processes. The file will be named sorted_integers.txt and will contain the sorted integers.
  """

  use GenServer

  @impl true
  def init({array_to_sort, n_workers}) do
    IO.puts("Master started with pid #{inspect(self())} \n")
    n_elements = length(array_to_sort)
    sorted_lists = []
    worker_pids = start_worker_processes(n_workers)

    start_initial_sorting(worker_pids, array_to_sort)

    {:ok, {sorted_lists, n_elements, worker_pids}}
  end

  def start_initial_sorting(available_workers, array_to_sort) do
    chunk_size = 500
    to_sort_lists = Enum.chunk_every(array_to_sort, chunk_size, chunk_size)
    worker_cycle = available_workers |> Stream.cycle()

    Enum.zip(worker_cycle, to_sort_lists)
    |> Enum.each(fn {worker_pid, list} -> DistributedWorker.sort(worker_pid, list) end)
  end

  def start_worker_processes(n_workers) do
    1..n_workers
    |> Enum.map(fn _ -> self() |> DistributedWorker.start_link() end)
    |> Enum.map(fn {:ok, pid} -> pid end)
  end

  @impl true
  def handle_call({:add_sorted_list, list}, from, state) do
    {sorted_lists, n_elements, worker_pids} = state

    sorted_lists = sorted_lists ++ [list]
    already_sorted_lists = length(sorted_lists)
    length_of_current_list = length(list)

    case {already_sorted_lists, length_of_current_list} do
      {1, ^n_elements} ->
        IO.puts("All elements sorted\n")
        to_write = list |> Enum.join("\n")
        File.write("sorted_integers.txt", to_write, [:write])
        sorted_lists = []
        {:stop, :normal, :ok, {sorted_lists, n_elements, worker_pids}}

      {2, _} ->
        IO.puts("Two lists sorted -> will call merge_sorted\n")
        {worker_pid, _} = from
        [first, second] = sorted_lists
        sorted_lists = []
        worker_pid |> DistributedWorker.merge_sorted({first, second})
        {:reply, sorted_lists, {sorted_lists, n_elements, worker_pids}}

      {_, _} ->
        {:reply, sorted_lists, {sorted_lists, n_elements, worker_pids}}
    end
  end

  @impl true
  def terminate(_reason, state) do
    IO.puts("Terminating DistributedMaster\n")
    {_, _, worker_pids} = state
    worker_pids |> Enum.each(fn pid -> DistributedWorker.finish(pid) end)
  end

  def add_sorted_list(pid, list) do
    GenServer.call(pid, {:add_sorted_list, list})
  end

  def start_link({array_to_sort, n_workers}) do
    GenServer.start_link(__MODULE__, {array_to_sort, n_workers})
  end
end
```

The `DistributedMaster` module is a [Genserver](https://hexdocs.pm/elixir/GenServer.html) that starts worker processes and sends lists to be sorted. It initially calls the worker processes to sort the lists and then merges the sorted lists. It implements 1 different call message:

- `{:add_sorted_list, list}` - Adds a sorted list to the list of sorted lists and calls for merging two lists when two lists were returned. Finally exits when one list of the size of the original is remaining.


Elixir genservers are lightweight [processes](https://hexdocs.pm/elixir/processes.html) that are used to manage state and handle messages. They are implemented using the `GenServer` module, which provides a set of callbacks that define the behaviour of the genserver. The `init` callback is called when the genserver is started and is responsible for initializing the genserver's state. The `handle_call` callback is called when a message is sent to the genserver using the `GenServer.call` function. The `terminate` callback is called when the genserver is terminated and is responsible for cleaning up any resources used by the genserver.

Processes receive messages in their `mailbox` which is a queue where messages are stored until they are processed. The `GenServer.call` function is used to send a message to a genserver and wait for a reply. The `GenServer.reply` function is used to send a reply to a message that was received by a genserver. The `GenServer.cast` function is used to send a message to a genserver without waiting for a reply.

In our case the distributed master process handles syncronous calls to add sorted lists and through [pattern matching](https://hexdocs.pm/elixir/pattern-matching.html) decides what to do with the sorted lists: 
- When two lists are sorted it casts a messsage `message_sorted` to the `DistributedWorker.merge_sorted` function to merge the two lists.
- When all lists are sorted it writes the sorted list to a file and returns `:stop` to terminate the genserver. If this is the case it also terminates all the child worker processes on the `terminate` callback.

The `DistributedWorker` module is a genserver that sorts a list of integers using the quicksort algorithm. The module implements 2 different cast messages:

- `{:sort, list}` - Sorts a list of integers using the quicksort algorithm.
- `{:merge_sorted, {list1, list2}}` - Merges two sorted lists into a single sorted list.

```elixir
defmodule DistributedWorker do
  @moduledoc """
  Distributed Worker module

  This module implements a GenServer that sorts a list of integers and merges two sorted lists.
  It implements 3 different cast messages:
  - {:sort, caller_pid, list} - Sorts the list and sends the sorted list to the caller_pid
  - {:merge_sorted_lists, caller_pid, list1, list2} - Merges the two sorted lists and sends the merged list to the caller_pid
  - :finish - Stops the GenServer
  """
  import Sorter, only: [merge_sorted_lists: 3, sort: 1]
  use GenServer

  def start_link(master_pid) do
    GenServer.start_link(__MODULE__, master_pid)
  end

  @impl true
  def init(master_pid) do
    IO.puts("Worker started with PID #{inspect(self())}\n")
    {:ok, master_pid}
  end

  @impl true
  def handle_cast({:sort, list}, state) do
    IO.puts("State #{inspect(state)}\n")
    master_pid = state
    sorted = sort(list)
    master_pid |> DistributedMaster.add_sorted_list(sorted)
    {:noreply, master_pid}
  end

  @impl true
  def handle_cast({:merge_sorted_lists, list1, list2}, state) do
    master_pid = state
    sorted = merge_sorted_lists([], list1, list2)
    master_pid |> DistributedMaster.add_sorted_list(sorted)
    {:noreply, master_pid}
  end

  @impl true
  def handle_cast(:finish, _state) do
    IO.puts("Stopping Worker\n")
    {:stop, :normal, %{}}
  end

  @impl true
  def terminate(_reason, _state) do
    IO.puts("Terminating Worker\n")
  end

  def finish(pid) do
    GenServer.cast(pid, :finish)
  end

  def sort(pid, list) do
    GenServer.cast(pid, {:sort, list})
  end

  def merge_sorted(pid, {list1, list2}) do
    GenServer.cast(pid, {:merge_sorted_lists, list1, list2})
  end
end
```

Notice that the DistributedWorker starts with a `master_pid` that is the pid of the `DistributedMaster` genserver. This is used to send messages to the master process to add the sorted lists. The `DistributedWorker` module is a genserver that sorts a list of integers using the quicksort algorithm. The module implements 2 different cast messages:
The process id is stored in the state via the `init` callback.


## Sorter module logic

The worker module imports the `Sorter` module `merge_sorted_lists/3` and `sort/1` functions.

```elixir
defmodule Sorter do
  @moduledoc """
  Module for sorting lists.

  It contains functionality for merging two sorted lists and sorting a list.
  Merging sorted lists complexity is O(m + n) where m and n are the lengths of the lists.
  Sorting a list complexity is O(n * log(n)) where n is the length of the list.
  """
  def merge_sorted_lists(sorted_acc, [], []) do
    sorted_acc
  end

  def merge_sorted_lists(sorted_acc, first, []) do
    sorted_acc ++ first
  end

  def merge_sorted_lists(sorted_acc, [], second) do
    sorted_acc ++ second
  end

  def merge_sorted_lists(sorted_acc, first, second) do
    [lhead | ltail] = first
    [rhead | rtail] = second

    {sorted_acc, first, second} =
      if lhead <= rhead do
        {sorted_acc ++ [lhead], ltail, second}
      else
        {sorted_acc ++ [rhead], first, rtail}
      end

    merge_sorted_lists(sorted_acc, first, second)
  end

  def sort(list) do
    Enum.sort(list)
  end
end
```

Notice the pattern-matching and recursion usage of the `merge_sorted_lists` function calls instead of explicit if-else statements that are checked 
during application runtime.
Instead we define a function for each possible case, to be efficiently compiled and let the BEAM VM decide which function to call at runtime.

