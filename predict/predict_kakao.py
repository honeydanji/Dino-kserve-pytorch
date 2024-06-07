# import requests
#
# host = "${HOST}"
# kbm_namespace = "${NAMESPACE}"
# username = "${USER_EMAIL}"
# password = "${USER_PASSWORD}"
# input_text_data = "Hello World!" # 테스트 문자열
#
# model_name = "torch-model"
# model_serv_name = "torchserve"
#
# session = requests.Session()
# \_kargs = {
# "verify": False
# }
#
# response = session.get("https://" + host, \*\*\_kargs)
#
# headers = {
# "Content-Type": "application/x-www-form-urlencoded",
# }
#
# session.post(response.url, headers=headers, data={"login": username, "password": password})
# session_cookie = session.cookies.get_dict()["authservice_session"]
# print(session_cookie)
#
# url = f"http://{host}/v1/models/{model_name}:predict"
# host = f"{model_serv_name}.{kbm_namespace}.{host}"
# print(url)
# print(host)
#
# session={'authservice_session': session_cookie}
# data = {"instances": [{"data": input_text_data}]}
#
# headers = {
# "Host": host,
# }
#
# x = requests.post(
# url=url,
# cookies=session,
# headers=headers,
# json=data
# )
#
# print(f"입력값: {data}")
# print(f"결과값: {x.text}")