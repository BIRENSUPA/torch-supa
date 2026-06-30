.. _native-api-torch_distributed_elastic:

torch.distributed.elastic
===============================
.. list-table::
   :header-rows: 1
   :widths: 60 15 25

   * - PyTorch API
     - 是否支持
     - 限制
   * - ``torch.distributed.elastic.agent.server.ElasticAgent``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.ElasticAgent.get_worker_group``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.ElasticAgent.run``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.WorkerSpec``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.WorkerSpec.get_entrypoint_name``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.WorkerState``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.WorkerState.is_running``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.Worker``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.WorkerGroup``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.SimpleElasticAgent``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.SimpleElasticAgent._assign_worker_ranks``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.SimpleElasticAgent._exit_barrier``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.SimpleElasticAgent._initialize_workers``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.SimpleElasticAgent._monitor_workers``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.SimpleElasticAgent._rendezvous``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.SimpleElasticAgent._restart_workers``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.SimpleElasticAgent._shutdown``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.SimpleElasticAgent._start_workers``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.SimpleElasticAgent._stop_workers``
     - 
     - 
   * - ``torch.distributed.elastic.agent.server.api.RunResult``
     - 
     - 
   * - ``torch.distributed.elastic.multiprocessing.start_processes``
     - 
     - 
   * - ``torch.distributed.elastic.multiprocessing.api.PContext``
     - 
     - 
   * - ``torch.distributed.elastic.multiprocessing.api.MultiprocessContext``
     - 
     - 
   * - ``torch.distributed.elastic.multiprocessing.api.SubprocessContext``
     - 
     - 
   * - ``torch.distributed.elastic.multiprocessing.api.RunProcsResult``
     - 
     - 
   * - ``torch.distributed.elastic.multiprocessing.errors.record``
     - 
     - 
   * - ``torch.distributed.elastic.multiprocessing.errors.ChildFailedError``
     - 
     - 
   * - ``torch.distributed.elastic.multiprocessing.errors.ErrorHandler``
     - 
     - 
   * - ``torch.distributed.elastic.multiprocessing.errors.ProcessFailure``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousParameters``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousParameters.get``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousParameters.get_as_bool``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousParameters.get_as_int``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousHandlerRegistry``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousHandlerRegistry.create_handler``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousHandlerRegistry.register``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousHandler``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousHandler.get_backend``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousHandler.get_run_id``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousHandler.is_closed``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousHandler.next_rendezvous``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousHandler.num_nodes_waiting``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousHandler.set_closed``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousHandler.shutdown``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousError``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousClosedError``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousTimeoutError``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousConnectionError``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.RendezvousStateError``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.dynamic_rendezvous.create_handler``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.dynamic_rendezvous.DynamicRendezvousHandler``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.dynamic_rendezvous.DynamicRendezvousHandler.from_backend``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.get_state``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.name``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.set_state``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.close``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.heartbeat``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.join``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.last_call``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.create_backend``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.get_state``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.name``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.set_state``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.create_backend``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.get_state``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.name``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.set_state``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.etcd_rendezvous.EtcdRendezvousHandler``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.etcd_store.EtcdStore``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.add``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.check``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.get``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.set``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.wait``
     - 
     - 
   * - ``torch.distributed.elastic.rendezvous.etcd_server.EtcdServer``
     - 
     - 
   * - ``torch.distributed.elastic.timer.configure``
     - 
     - 
   * - ``torch.distributed.elastic.timer.expires``
     - 
     - 
   * - ``torch.distributed.elastic.timer.LocalTimerServer``
     - 
     - 
   * - ``torch.distributed.elastic.timer.LocalTimerClient``
     - 
     - 
   * - ``torch.distributed.elastic.timer.FileTimerServer``
     - 
     - 
   * - ``torch.distributed.elastic.timer.FileTimerClient``
     - 
     - 
   * - ``torch.distributed.elastic.timer.TimerRequest``
     - 
     - 
   * - ``torch.distributed.elastic.timer.TimerServer``
     - 
     - 
   * - ``torch.distributed.elastic.timer.TimerServer.clear_timers``
     - 
     - 
   * - ``torch.distributed.elastic.timer.TimerServer.get_expired_timers``
     - 
     - 
   * - ``torch.distributed.elastic.timer.TimerServer.register_timers``
     - 
     - 
   * - ``torch.distributed.elastic.timer.TimerClient``
     - 
     - 
   * - ``torch.distributed.elastic.timer.TimerClient.acquire``
     - 
     - 
   * - ``torch.distributed.elastic.timer.TimerClient.release``
     - 
     - 
   * - ``torch.distributed.elastic.metrics.api.MetricHandler``
     - 
     - 
   * - ``torch.distributed.elastic.metrics.api.ConsoleMetricHandler``
     - 
     - 
   * - ``torch.distributed.elastic.metrics.api.NullMetricHandler``
     - 
     - 
   * - ``torch.distributed.elastic.metrics.configure``
     - 
     - 
   * - ``torch.distributed.elastic.metrics.prof``
     - 
     - 
   * - ``torch.distributed.elastic.metrics.put_metric``
     - 
     - 
   * - ``torch.distributed.elastic.events.record``
     - 
     - 
   * - ``torch.distributed.elastic.events.get_logging_handler``
     - 
     - 
   * - ``torch.distributed.elastic.events.api.Event``
     - 
     - 
   * - ``torch.distributed.elastic.events.api.EventSource``
     - 
     - 
   * - ``torch.distributed.elastic.events.api.EventMetadataValue``
     - 
     - 
