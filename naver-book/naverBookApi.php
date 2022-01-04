<?php
class Book
{
        private $key = "c1b406b32dbbbbeee5f2a36ddc14067f"; // 사용자가 발급받은 오픈 API 키
        private $searchUrl = "http://openapi.naver.com/search"; // 오픈 API 호출 URL
        private $target = "book";

        /**
         * API 결과를 받아오기 위하여 오픈 API 서버에 Request를 하고 결과를 XML Object로 반환하는 메서드
         * @return object
         */
        private function query($query)
        {
                $url = sprintf("%s?query=%s&target=%s&key=%s", $this->searchUrl, $query, $this->target, $this->key);
                $data =file_get_contents($url);
                $xml = simplexml_load_string($data);

                return $xml;
        }

        /**
         * API의 결과를 Json으로 encode하려 반환하는 메서드
         * XML을 직접 parsing하지 않고 json으로 변환하여 반환한다.
         */
        public function getBookSearch($query)
        {       $xml = $this->query($query);
                $result = json_encode($xml);

                return $result;
        }
}
?>
<?php
class Book
{
        private $key = "c1b406b32dbbbbeee5f2a36ddc14067f"; // 사용자가 발급받은 오픈 API 키
        private $searchUrl = "http://openapi.naver.com/search"; // 오픈 API 호출 URL
        private $target = "book";

        /**
         * API 결과를 받아오기 위하여 오픈 API 서버에 Request를 하고 결과를 XML Object로 반환하는 메서드
         * @return object
         */
        private function query($query)
        {
                $url = sprintf("%s?query=%s&target=%s&key=%s", $this->searchUrl, $query, $this->target, $this->key);
                $data =file_get_contents($url);
                $xml = simplexml_load_string($data);

                return $xml;
        }

        /**
         * API의 결과를 Json으로 encode하려 반환하는 메서드
         * XML을 직접 parsing하지 않고 json으로 변환하여 반환한다.
         */
        public function getBookSearch($query)
        {       $xml = $this->query($query);
                $result = json_encode($xml);

                return $result;
        }
}
?>